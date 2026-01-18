import os

# These flags are optional. In some cases, you can see a speedup if you are using GPUs,
# but they are mostly experimental, and have a 50-50 chance of working.
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true "
)
os.environ.update(
    {
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    }
)


import warnings
import logging
import time
from pathlib import Path
from functools import partial

import jax
import optax
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
from fineweb_dataloader import DataPreloader, BOSFinder, _load_data_shard
from config import ShardingRules, Config, BATCH_AXIS_NAME

logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


def compute_loss(params, x_batch, y_batch, freqs):
    # Logits are already in fp32. If not, ensure to cast them.
    logits = forward(params, x_batch, freqs)
    loss = jnp.mean(
        optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=y_batch
        )
    )
    return loss, logits


@partial(jax.jit, static_argnames=("optim",), donate_argnums=(0, 4))
def train_step_accum(params, x_batch, y_batch, freqs, optim_state, optim):
    (loss, logits), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        params, x_batch, y_batch, freqs
    )
    _, optim_state = optim.update(grads, optim_state, params)
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


def set_layerwise_lr(
    params,
    *,
    d_model: int,
    other_peak_lr: float,  # peak LR for non-(embed/lm_head)
    other_min_lr: float,  # end/min LR for non-(embed/lm_head)
    total_train_steps: int,
    warmup_steps: int = 30,
    weight_decay: float = 0.001,
    b1: float = 0.9,
    b2: float = 0.95,
    embedding_lr: float = 3e-3,
    unembedding_lr: float = 3e-4,
):
    """
    Parameter groups:
      - params.embed      -> AdamW with nanochat embedding LR (scaled by (d_model/768)^-0.5)
      - params.lm_head    -> AdamW with nanochat unembedding LR (scaled by (d_model/768)^-0.5)
      - everything else   -> AdamW with warmup+cosine schedule

    Returns: (optax_transform, param_labels)
    """

    # nanochat's width scaling for AdamW groups: (d_model / 768) ** -0.5
    # This is slowing down learning. TODO: Debug and use it later for scaling lr
    dmodel_lr_scale = (d_model / 768.0) ** -0.5  # noqa

    emb_lr = embedding_lr * other_peak_lr
    unemb_lr = 1.0 * other_peak_lr

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

    def make_adamw(lr_schedule):
        return optax.adamw(
            learning_rate=lr_schedule,
            b1=b1,
            b2=b2,
            weight_decay=weight_decay,
            mu_dtype=jnp.float32,
        )

    optim = optax.multi_transform(
        {
            "embed": make_adamw(schedules["embed"]),
            "lm_head": make_adamw(schedules["lm_head"]),
            "other": make_adamw(schedules["other"]),
        },
        param_labels,
    )

    return optim


# Get the mesh, sharding rules, amd the config
devices = np.array(jax.devices())
print("Number of devices found:", len(devices))
mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
cfg = Config(mesh=mesh, rules=sharding_rules)

# Load the model
print("Building GPT model based on the config...")
model = GPT.init(jax.random.PRNGKey(0), cfg)
print("Model built successfully!")


per_device_bsz = cfg.hparams.per_device_batch_size
bsz = per_device_bsz * len(devices)
seqlen = cfg.model.seqlen
head_dim = cfg.model.attn.head_dim
data_sharding = logical_to_sharding(("batch",), cfg.mesh, cfg.rules)

max_lr = cfg.hparams.max_lr
min_lr = 0.01 * max_lr
warmup_steps = cfg.hparams.warmup_steps
desired_batch_size = cfg.hparams.desired_batch_size

# Turn off grad_accum for now. MultiStep is behaving in a bad way when branched
# out for efficiency. Pretty sure, I am doing something stupid, but turn it off
# for now. It is necessary for squeezing out every ounce of GPU though.
grad_accum_steps = 1  # max(4, desired_batch_size // (bsz * seqlen))

b1 = cfg.hparams.b1
b2 = cfg.hparams.b2
weight_decay = cfg.hparams.weight_decay
grad_clip_norm = cfg.hparams.grad_clip_norm
total_train_steps = cfg.hparams.total_train_steps
max_checkpoints_to_keep = cfg.ckpt_cfg.max_checkpoints_to_keep
checkpoint_save_steps = cfg.ckpt_cfg.checkpoint_save_steps


# Optimizer
optim = optax.chain(
    optax.clip_by_global_norm(grad_clip_norm),
    set_layerwise_lr(
        model,
        d_model=cfg.model.d_emb,
        other_peak_lr=max_lr,
        other_min_lr=min_lr,
        total_train_steps=total_train_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    ),
)

# TODO: Turn ot on after debugging
# optim = optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)

optim_state = optim.init(model)

# Checkpointing
ckpt_path = Path(cfg.ckpt_cfg.save_ckpt_dir)
options = ocp.CheckpointManagerOptions(
    max_to_keep=max_checkpoints_to_keep, save_interval_steps=checkpoint_save_steps
)
handlers = {
    "params": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
    "optim_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
}

mngr = ocp.CheckpointManager(ckpt_path, handlers, options=options)

print("\n", "-" * 75, "\n")
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


best_loss = float("inf")
patience = cfg.hparams.es_patience
patience_counter = 0
best_step = 0
last_checkpoint_step = 0
val_interval = 50
last_val_loss = None
num_epochs = 1
worker_count = 0  # Number of parallel workers for data loading
worker_buffer_size = 1  # Prefetch buffer size per worker

train_files = list(Path(cfg.data_dir).glob("*train*.bin"))
val_files = list(Path(cfg.data_dir).glob("*val*.bin"))
num_train_files = len(train_files)
num_val_files = len(val_files)
print("\nNumber of train files found: ", num_train_files)
print("Number of validation files found: ", num_val_files)

# Load first shard synchronously
train_iter = iter(train_files)
val_iter = iter(val_files)


train_tokens = _load_data_shard(next(train_iter))
train_finder = BOSFinder(train_tokens)
train_preloader = DataPreloader(train_iter, bsz)

val_tokens = _load_data_shard(next(val_iter))
val_finder = BOSFinder(val_tokens)
val_preloader = DataPreloader(val_iter, bsz)

step = 0
print("Starting training (the first step will take some time for compilation...)\n")
training_complete = False

# Training loop with explicit counter
for shard_idx in range(num_train_files):
    if training_complete:
        break
    # Start preloading next shard if not on last shard
    if shard_idx < num_train_files - 1:
        train_preloader.start()

    shard_processed_fully = False
    print(f"\n=== Processing Shard {shard_idx + 1}/{num_train_files} ===")

    while not shard_processed_fully:
        try:
            start = time.time()
            train_step_loss = 0.0
            for micro_step in range(grad_accum_steps):
                seq_starts, seq_ends = train_finder.next_batch(bsz, seqlen)
                # TODO: This is bad, and is a bottleneck. Make it more efficient
                buf = (
                    np.concatenate(
                        [train_tokens[i:j] for i, j in zip(seq_starts, seq_ends)]
                    )
                    .reshape(bsz, seqlen + 1)
                    .astype(np.int32)
                )
                x = buf[:, :-1]
                y = buf[:, 1:]
                x = jax.device_put(x, data_sharding)
                y = jax.device_put(y, data_sharding)
                if micro_step < grad_accum_steps - 1:
                    model, loss, optim_state = train_step_accum(
                        model, x, y, freqs, optim_state, optim
                    )
                    train_step_loss += loss
                else:
                    model, loss, optim_state = train_step(
                        model, x, y, freqs, optim_state, optim
                    )
                    train_step_loss += loss

            avg_train_loss = train_step_loss / grad_accum_steps
            step += 1

            # Block for accurate timing
            jax.block_until_ready(avg_train_loss)

            end = time.time()
            dt = end - start

            tokens_processed = bsz * seqlen * grad_accum_steps
            tokens_per_sec = int(tokens_processed / dt)

            improved = avg_train_loss < best_loss
            if improved:
                best_loss = avg_train_loss
                best_step = step
                patience_counter = 0
            else:
                patience_counter += 1

            if improved or (step % options.save_interval_steps) == 0:
                mngr.save(
                    step,
                    args=ocp.args.Composite(
                        params=ocp.args.PyTreeSave(model),
                        optim_state=ocp.args.PyTreeSave(optim_state),
                    ),
                )
            print(
                f"Step: [{str(step).zfill(len(str(total_train_steps)))}/{total_train_steps}] | loss: {avg_train_loss:8.4f} | Time taken: {dt:8.2f} s | Tokens processed/s: {tokens_per_sec:>9,}"
            )

            # Check stopping conditions
            if patience_counter > patience:
                print(
                    f"\nEarly stopping triggered! No improvement for {patience} steps."
                )
                print(f"Best loss: {best_loss:.4f} at step {best_step}")
                mngr.wait_until_finished()
                training_complete = True  # ← Set flag
                break  # ← Break inner loop

            if step >= total_train_steps:
                print(f"\nReached maximum training steps: {total_train_steps}")
                mngr.wait_until_finished()
                print("Finished checkpointing! Cleaned.")
                training_complete = True  # ← Set flag
                break  # ← Break inner loop

        except StopIteration:
            print(f"{shard_idx + 1}/{num_train_files} shard processed fully.")
            shard_processed_fully = True

    # Load next shard if available
    if shard_idx < num_train_files - 1:
        train_tokens, train_finder = train_preloader.get()

    if step >= total_train_steps or training_complete:
        mngr.wait_until_finished()
        print("Finished checkpointing! Cleaned.")
        break
