import os
import warnings
import logging
import time
from pathlib import Path
from functools import partial

from fineweb_dataloader import make_grain_shard_loader, BOSFinder

import jax
import optax
import grain
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.sharding import Mesh
from jax.sharding import set_mesh
from jax.tree_util import GetAttrKey, SequenceKey, DictKey


from model import count_params
from model import precompute_frequencies
from model import GPT, forward
from utils import logical_to_sharding
from config import ShardingRules, Config, BATCH_AXIS_NAME


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


policy = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
forward_remat = jax.checkpoint(forward, policy=policy)

def compute_loss(params, x_batch, y_batch, freqs):
    logits = forward_remat(params, x_batch, freqs)
    loss = jnp.mean(
        optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=y_batch
        )
    )
    return loss, logits

@partial(jax.jit, static_argnames=("optim", "grad_accum_steps"), donate_argnums=(0, 1, 4))
def train_step_accum(params, x_batch, y_batch, freqs, optim_state, optim, grad_accum_steps):
    # x_batch, y_batch: [grad_accum_steps, bsz, seqlen]
    def body(carry, xy):
        p, gsum, lsum = carry
        xb, yb = xy
        (loss, _), g = jax.value_and_grad(compute_loss, has_aux=True)(p, xb, yb, freqs)
        gsum = jax.tree_util.tree_map(lambda a, b: a + b, gsum, g)
        lsum = lsum + loss
        return (p, gsum, lsum), None

    g0 = jax.tree_util.tree_map(jnp.zeros_like, params)
    carry0 = (params, g0, jnp.array(0.0, dtype=jnp.result_type(0.0)))
    (p, gsum, lsum), _ = jax.lax.scan(body, carry0, (x_batch, y_batch), length=grad_accum_steps)

    steps = grad_accum_steps
    gsum = jax.tree_util.tree_map(lambda g: g / steps, gsum)
    loss = lsum / steps

    updates, optim_state = optim.update(gsum, optim_state, p)
    params = optax.apply_updates(p, updates)
    return params, loss, optim_state



@partial(jax.jit, static_argnames=("optim",), donate_argnums=(0, 4))
def train_step(params, x_batch, y_batch, freqs, optim_state, optim):
    (loss, logits), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        params, x_batch, y_batch, freqs
    )
    updates, optim_state = optim.update(grads, optim_state, params)
    updated_params = optax.apply_updates(params, updates)
    return updated_params, loss, optim_state


@jax.jit
def val_step(params, x_batch, y_batch, freqs):
    loss, _ = compute_loss(params, x_batch, y_batch, freqs)
    return loss


def line(label, value, comma=False, label_w=30, colon_w=2, value_w=20):
    fmt = f">{value_w}," if comma else f">{value_w}"
    return f"{label:<{label_w}}{':':<{colon_w}}{value:{fmt}}"


def build_optimizer(
    params,
    *,
    d_model: int,                    
    other_peak_lr: float,           
    other_min_lr: float,            
    total_train_steps: int,
    warmup_steps: int = 30,
    b1: float = 0.9,                
    b2: float = 0.95,           
    embedding_lr: float = 3e-3,
    unembedding_lr: float = 3e-4,
    use_muon=True

):
    """
    Parameter groups:
      - params.embed      -> AdamW with nanochat embedding LR (scaled by (d_model/768)^-0.5)
      - params.lm_head    -> AdamW with nanochat unembedding LR (scaled by (d_model/768)^-0.5)
      - everything else   -> AdamW with warmup+cosine schedule

    Returns: (optax_transform, param_labels)
    """

    # nanochat's width scaling for AdamW groups: (d_model / 768) ** -0.5
    dmodel_lr_scale = (d_model / 768.0) ** -0.5

    # emb_lr = embedding_lr * dmodel_lr_scale
    # unemb_lr = unembedding_lr * dmodel_lr_scale
    emb_lr = embedding_lr * other_peak_lr
    unemb_lr = 1.0 * other_peak_lr

    if use_muon:
        print("Using Muon Optimizer!")
        other_peak_lr = max(other_peak_lr, 2e-2)


    schedules = {
        "embed": optax.constant_schedule(emb_lr),
        "lm_head": optax.constant_schedule(unemb_lr),
        "other": optax.warmup_cosine_decay_schedule(
            init_value=other_min_lr,
            peak_value=other_peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=max(1, total_train_steps - warmup_steps),
            end_value=other_min_lr,
        ),
    }

    def _path_names(path):
        out = []
        for k in path:
            if isinstance(k, GetAttrKey):
                out.append(k.name)
            elif isinstance(k, SequenceKey):
                out.append(str(k.idx))
            elif isinstance(k, DictKey):
                out.append(str(k.key))
            else:
                out.append(str(k))
        return out

    def label_fn(path, leaf):
        # Top-level fields in your GPT pytree: embed, blocks, lm_head
        names = _path_names(path)
        top = names[0] if names else ""
        if top == "embed":
            return "embed"
        if top == "lm_head":
            return "lm_head"
        return "other"

    param_labels = jax.tree_util.tree_map_with_path(label_fn, params)

    def make_adamw(lr_schedule, weight_decay=0.0):
        return optax.adamw(
            learning_rate=lr_schedule,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            mu_dtype=jnp.float32
        )

    def make_weight_dim_nums(p):
        def choose(x):
            s = getattr(x, "shape", None)
            if s is None:
                return None
            if len(s) == 2:
                return optax.contrib.MuonDimensionNumbers((0,), (1,))
            if len(s) == 3:
                if s[-1] == d_model:       # wo: (heads, head_dim, d_model)
                    return optax.contrib.MuonDimensionNumbers((1,), (2,))
                return optax.contrib.MuonDimensionNumbers((0,), (2,))  # wq/wk/wv: batch=heads
            return None
        return jax.tree_util.tree_map(choose, p)

    muon_weight_dim_nums = make_weight_dim_nums(params)

    def weight_decay_mask_fn(p):
        def keep(x):
            s = getattr(x, "shape", None)
            return s is not None and len(s) >= 2
        return jax.tree_util.tree_map(keep, p)

    muon_wd_mask = weight_decay_mask_fn(params)


    def make_muon(lr_schedule, weight_decay=0.0):
        return optax.contrib.muon(
            learning_rate=lr_schedule,
            ns_coeffs=(3.4445, -4.775, 2.0315), 
            ns_steps=3,#5,
            beta=b2,
            eps=1e-8,
            # weight_decay=weight_decay,
            weight_decay=weight_decay,
            weight_decay_mask=muon_wd_mask,#wd_mask_fn,
            mu_dtype=jnp.float32,
            nesterov=True,
            adaptive=False,
            adam_b1=b1,
            adam_b2=b2,
            adam_eps_root=0.0,
            adam_weight_decay=weight_decay,
            muon_weight_dimension_numbers=muon_weight_dim_nums,#muon_dims_fn,
            consistent_rms=None
        )

    # then change only the "other" branch of the multi_transform to use Muon
    tx = optax.multi_transform(
        {
            "embed": make_adamw(schedules["embed"]),
            "lm_head": make_adamw(schedules["lm_head"]),
            "other": make_muon(schedules["other"], weight_decay=0.001) if use_muon else make_adamw(schedules["other"], weight_decay=0.001),
        },
        param_labels,
    )
    return tx


def get_next_batch(starts, ends, bsz, seqlen, tokens, data_sharding, transfer_to_device=True, out_tokens_u16=None):
    # out_tokens_u16: (bsz, seqlen+1) uint16 workspace; if None, allocate once here (compat fallback)
    if out_tokens_u16 is None:
        out_tokens_u16 = np.empty((bsz, seqlen + 1), dtype=np.uint16)

    ptr = 0
    for i, j in zip(starts, ends):
        n = j - i
        row = ptr // (seqlen + 1)
        col = ptr % (seqlen + 1)
        out_tokens_u16[row, col:col + n] = tokens[i:j]
        ptr += n

    # Optional device path: single H2D for tokens; slice into x/y on device
    if transfer_to_device:
        dev_tokens_u16 = jax.device_put(out_tokens_u16, data_sharding)
        dev_tokens_i32 = jax.lax.convert_element_type(dev_tokens_u16, jnp.int32)
        x = dev_tokens_i32[:, :-1]
        y = dev_tokens_i32[:, 1:]
        return x, y
    else:
        return None, None


def main():

    # Get the mesh, sharding rules, amd the config
    devices = np.array(jax.devices())
    print("Number of devices found:", len(devices))
    mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
    sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
    cfg = Config(mesh=mesh, rules=sharding_rules)

    train_files = list(Path(cfg.data_dir).glob("*train*.bin"))
    val_files = list(Path(cfg.data_dir).glob("*val*.bin"))
    num_train_files = len(train_files)
    num_val_files = len(val_files)
    print("\nNumber of train files found: ", num_train_files)
    print("Number of validation files found: ", num_val_files)
    
    train_dl = make_grain_shard_loader(train_files, prefetch=16, num_threads=32, prefetch_buffer_size=256)
    val_dl = make_grain_shard_loader(val_files, prefetch=0, num_threads=16, prefetch_buffer_size=256)
    train_iter = iter(train_dl)


    per_device_bsz = cfg.hparams.per_device_batch_size
    bsz = per_device_bsz * len(devices)
    seqlen = cfg.model.seqlen
    head_dim = cfg.model.attn.head_dim
    data_sharding = logical_to_sharding(("batch",), cfg.mesh, cfg.rules)

    max_lr = cfg.hparams.max_lr
    min_lr = 0.01 * max_lr
    warmup_steps = cfg.hparams.warmup_steps
    desired_batch_size = cfg.hparams.desired_batch_size
    grad_accum_steps = max(4, desired_batch_size // (bsz * seqlen))
    
    b1 = cfg.hparams.b1
    b2 = cfg.hparams.b2
    weight_decay = cfg.hparams.weight_decay
    grad_clip_norm = cfg.hparams.grad_clip_norm
    
    total_train_steps = cfg.hparams.total_train_steps
    max_checkpoints_to_keep = cfg.ckpt_cfg.max_checkpoints_to_keep
    checkpoint_save_steps = cfg.ckpt_cfg.checkpoint_save_steps


    # Load the model
    print("Building GPT model based on the config...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    print("Model built successfully!")


    # Optimizer
    optim = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        build_optimizer(
            model,
            d_model=cfg.model.d_emb,
            other_peak_lr=max_lr,
            other_min_lr=min_lr,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
    )


    # No grad accum for now
    if grad_accum_steps > 1:
        print("Applying grad accum schedule to the optimizer...")
        # optim = optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)
    optim_state = optim.init(model)


    print("")
    print("-" * 75)
    print("")


    print(line("Number of trainable params: ", count_params(model), comma=True))
    print(line("Sequence length per sample", seqlen))
    print(line("Per device batch size", per_device_bsz))
    print(line("Total batch size", bsz))
    print(line("Grad accumulation steps", grad_accum_steps))
    print()
    print(line("LR (min, max)", str((min_lr, max_lr))))
    print(line("Warmup steps", warmup_steps))
    print(line("Weight decay", weight_decay), "\n")
    print("-" * 75)


    # Compute the frequencies
    positions = jnp.arange(seqlen)[None, :]
    with set_mesh(cfg.mesh):
        freqs = precompute_frequencies(positions=positions, features=head_dim)

    num_shards_used = 0
    step = 0
    print("Starting training (the first step will take some time for compilation...)\n")
    profiling_complete = False
    profile_start_step = 5
    profile_end_step = 20

    data_accum_sharding = logical_to_sharding((None, "batch", None), cfg.mesh, cfg.rules)
    # batch_x = np.zeros((grad_accum_steps, bsz, seqlen), dtype=np.int32)
    # batch_y = np.zeros((grad_accum_steps, bsz, seqlen), dtype=np.int32)
    
    batch_tokens0 = np.empty((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)
    batch_tokens1 = np.empty((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)
    batch_x = batch_tokens0[:, :, :-1]  # host view (uint16); kept for name compatibility
    batch_y = batch_tokens0[:, :, 1:]   # host view (uint16); kept for name compatibility



    # Training loop with explicit counter
    for shard in train_iter:

        if profiling_complete:
            break
        
        tokens = shard["tokens"]        # SharedMemoryArray(uint16, shape=[num_tokens])
        bos_idx = shard["bos_idx"]      # np.ndarray of BOS positions
        size = shard["size"]
        shard_name = Path(shard["path"]).name

        try:

            # Optional: reuse your BOSFinder but skip its thread methods.
            bf = BOSFinder(tokens)
            bf.bos_idx = bos_idx
            bf.size = size
            
            shard_processed_fully = False
            # NEW: build the static index once per shard (on-demand)
            num_batches_in_shard = bf.build(bsz, seqlen)
            print(f"\n=== Processing Shard: {num_shards_used} with name: {shard_name}", end=" | ")
            print(f"Indexed {num_batches_in_shard} batches ===")
            
            while not shard_processed_fully:
                try:
                    if step < profile_start_step:
                        start = time.time()
                        for micro_step in range(grad_accum_steps):
                            starts, ends = bf.next_batch(bsz, seqlen)
                            get_next_batch(starts, ends, bsz, seqlen, tokens, data_accum_sharding, transfer_to_device=False, out_tokens_u16=batch_tokens0[micro_step])

                        # One H2D for stacked tokens; cast + slice on device
                        stacked_tokens = jax.device_put(batch_tokens0, data_accum_sharding)
                        stacked_tokens_i32 = jax.lax.convert_element_type(stacked_tokens, jnp.int32)
                        stacked_x = stacked_tokens_i32[:, :, :-1]
                        stacked_y = stacked_tokens_i32[:, :, 1:]
                        
                        model, loss, optim_state = train_step_accum(model, stacked_x, stacked_y, freqs, optim_state, optim, grad_accum_steps)
                        avg_train_loss = loss

                        # Block for accurate timing
                        jax.block_until_ready(avg_train_loss)
                        end = time.time()
                        dt = end - start
                        tokens_processed = bsz * seqlen * grad_accum_steps
                        tokens_per_sec = int(tokens_processed / dt)
                        print(f"Step: [{str(step).zfill(len(str(total_train_steps)))}/{total_train_steps}] | loss: {avg_train_loss:8.4f} | Time taken: {dt:5.2f} s | Tokens processed/s: {tokens_per_sec:>9,}")
                        step += 1
                    else:
                        logdir = "/home/ubuntu/nanochat/jaxnano/final_checks/profile-data/"
                        jax.profiler.start_trace(logdir)

                        with jax.profiler.TraceAnnotation("host_stack"):
                            for micro_step in range(grad_accum_steps):
                                starts, ends = bf.next_batch(bsz, seqlen)
                                get_next_batch(starts, ends, bsz, seqlen, tokens, data_accum_sharding,
                                            transfer_to_device=False, out_tokens_u16=batch_tokens0[micro_step])

                        with jax.profiler.TraceAnnotation("h2d"):
                            stacked_tokens_cur = jax.device_put(batch_tokens0, data_accum_sharding)
                            stacked_tokens_i32 = jax.lax.convert_element_type(stacked_tokens_cur, jnp.int32)

                        for profile_step in range(profile_start_step, profile_end_step):
                            with jax.profiler.StepTraceAnnotation("train", step_num=profile_step):
                                # Launch compute for current step (cast + slice on device)
                                with jax.profiler.TraceAnnotation("train_step_accum"):
                                    stacked_x = stacked_tokens_i32[:, :, :-1]
                                    stacked_y = stacked_tokens_i32[:, :, 1:]
                                    model, loss, optim_state = train_step_accum(
                                        model, stacked_x, stacked_y, freqs, optim_state, optim, grad_accum_steps
                                    )

                                # While compute runs, prepare the next step on host
                                with jax.profiler.TraceAnnotation("host_stack"):
                                    for micro_step in range(grad_accum_steps):
                                        starts, ends = bf.next_batch(bsz, seqlen)
                                        get_next_batch(starts, ends, bsz, seqlen, tokens, data_accum_sharding,
                                                    transfer_to_device=False, out_tokens_u16=batch_tokens1[micro_step])

                                # Queue the next H2D before we block on this step's loss
                                with jax.profiler.TraceAnnotation("h2d"):
                                    stacked_tokens_next = jax.device_put(batch_tokens1, data_accum_sharding)

                                # Synchronize for accurate timing after next H2D is queued
                                jax.block_until_ready(loss)

                                # Rotate buffers for the next iteration
                                batch_tokens0, batch_tokens1 = batch_tokens1, batch_tokens0
                                stacked_tokens_cur = stacked_tokens_next
                        jax.profiler.stop_trace()
                        profiling_complete = True
                        break

                except StopIteration:
                    # Once we have trained on one shard, let's validate the performance as well
                    num_shards_used +=1
                    print(f"\nShard exhuasted. Number of shards consumed till now: {num_shards_used}. Scoring model performance on vdalidation data...")
                    shard_processed_fully = True
        finally:    
            tokens.unlink_on_del()

if __name__ == "__main__":
    main()
