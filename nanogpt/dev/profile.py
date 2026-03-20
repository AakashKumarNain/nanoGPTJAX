import os

# Set some GPU FLAGS
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["NCCL_NVLS_ENABLE"] = "1"
os.environ.update(
    {
        "NCCL_LL128_BUFFSIZE": "-2",
        "NCCL_LL_BUFFSIZE": "-2",
        "NCCL_PROTO": "SIMPLE,LL,LL128",
    }
)
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_pipelined_all_reduce=true "
    "--xla_gpu_enable_pipelined_all_gather=true "
    "--xla_gpu_enable_pipelined_reduce_scatter=true "
#     "--xla_gpu_enable_while_loop_double_buffering=true "
#     "--xla_gpu_enable_pipelined_p2p=true "
#     "--xla_gpu_collective_permute_decomposer_threshold=1024 "
)
import warnings
import logging
import time
import dataclasses
from pathlib import Path
from functools import partial

import jax

jax.config.update("jax_optimization_level", "O1")

import optax
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import set_mesh

from model import count_params
from model import precompute_frequencies
from model import GPT, forward
from utils import logical_to_sharding
from optim import build_optimizer
from custom_loss import (
    DEFAULT_CHUNK_SIZE,
    chunked_softmax_cross_entropy_with_integer_labels,
)
from config import (
    Config,
    ShardingRules,
    BATCH_AXIS_NAME,
    TENSOR_ONLY_AXIS_NAME,
)
from fineweb_dataloader import make_grain_shard_loader, BOSFinder


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


PROFILE_LOSS_IMPL = os.environ.get("PROFILE_LOSS_IMPL", "chunked").lower()
PROFILE_CE_CHUNK_SIZE = int(
    os.environ.get("PROFILE_CE_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))
)


@dataclasses.dataclass(frozen=True)
class RuntimeShardings:
    embed_act: jax.sharding.Sharding
    logits: jax.sharding.Sharding


def compute_loss(
    params,
    x_batch,
    y_batch,
    segment_ids,
    freqs,
    loss_mask,
    runtime_shardings,
):
    logits = forward(
        params,
        x_batch,
        segment_ids,
        freqs,
        runtime_shardings.embed_act,
        runtime_shardings.logits,
    )
    if PROFILE_LOSS_IMPL == "chunked":
        return chunked_softmax_cross_entropy_with_integer_labels(
            logits,
            y_batch,
            loss_mask,
            chunk_size=PROFILE_CE_CHUNK_SIZE,
        )
    elif PROFILE_LOSS_IMPL == "dense":
        if loss_mask is not None:
            per_token_loss = optax.losses.softmax_cross_entropy_with_integer_labels(
                logits=logits,
                labels=y_batch,
                where=loss_mask,
            )
            return jnp.sum(per_token_loss) / jnp.maximum(jnp.sum(loss_mask), 1.0)
        else:
            return jnp.mean(
                optax.losses.softmax_cross_entropy_with_integer_labels(
                    logits=logits, labels=y_batch
                )
            )
    else:
        raise ValueError(
            f"Unsupported PROFILE_LOSS_IMPL={PROFILE_LOSS_IMPL!r}. Use 'chunked' or 'dense'."
        )


@partial(
    jax.jit,
    static_argnames=("optim", "grad_accum_steps", "runtime_shardings"),
    donate_argnums=(0, 1, 3, 4, 5),
)
def train_step_accum(
    params,
    x_batch,
    y_batch,
    segment_ids,
    freqs,
    optim_state,
    optim,
    grad_accum_steps,
    runtime_shardings,
):
    def body(carry, xy):
        param, opt_state, lsum = carry
        xb, yb = xy
        loss, grad = jax.value_and_grad(compute_loss)(
            param, xb, yb, segment_ids, freqs, None, runtime_shardings
        )

        # MultiSteps accumulates grad internally and returns a zero-tree update on
        # every micro-step except the last, where it emits the real update.
        updates, new_opt_state = optim.update(grad, opt_state, param)
        new_param = optax.apply_updates(param, updates)
        return (new_param, new_opt_state, lsum + loss), None

    carry0 = (params, optim_state, jnp.array(0.0, dtype=jnp.result_type(0.0)))
    (params, optim_state, lsum), _ = jax.lax.scan(
        body, carry0, (x_batch, y_batch), length=grad_accum_steps
    )
    loss = lsum / grad_accum_steps
    return params, loss, optim_state


@partial(
    jax.jit,
    static_argnames=("optim", "runtime_shardings"),
    donate_argnums=(0, 1, 3, 4, 5),
)
def train_step(
    params,
    x_batch,
    y_batch,
    segment_ids,
    freqs,
    optim_state,
    optim,
    runtime_shardings,
):
    loss, grads = jax.value_and_grad(compute_loss)(
        params,
        x_batch,
        y_batch,
        segment_ids,
        freqs,
        None,
        runtime_shardings,
    )
    updates, optim_state = optim.update(grads, optim_state, params)
    updated_params = optax.apply_updates(params, updates)
    return updated_params, loss, optim_state


def line(label, value, comma=False, label_w=30, colon_w=2, value_w=20):
    fmt = f">{value_w}," if comma else f">{value_w}"
    return f"{label:<{label_w}}{':':<{colon_w}}{value:{fmt}}"


def model_run_name(cfg):
    return (
        f"{cfg.model.attn_type}"
        f"_L{cfg.model.num_layers}"
        f"_D{cfg.model.d_emb}"
        f"_Q{cfg.model.q_heads}"
        f"_KV{cfg.model.kv_heads}"
        f"_H{cfg.model.attn.head_dim}"
        f"_T{cfg.model.seqlen}"
        f"_V{cfg.model.vocab_size}"
    )


def get_next_batch(
    starts,
    ends,
    bsz,
    seqlen,
    tokens,
    data_sharding,
    buf_u16,
    transfer_to_device=False,
    create_new_buf=False,
):
    """Gathers batches of input-labels pairs."""
    if buf_u16 is None and create_new_buf:
        buf_u16 = np.empty((bsz, seqlen + 1), dtype=np.uint16)

    ptr = 0
    for i, j in zip(starts, ends):
        n = j - i
        row = ptr // (seqlen + 1)
        col = ptr % (seqlen + 1)
        buf_u16[row, col : col + n] = tokens[i:j]
        ptr += n

    if not create_new_buf:
        return None
    else:
        if transfer_to_device:
            x = jax.device_put(buf_u16[:, :-1], data_sharding)
            y = jax.device_put(buf_u16[:, 1:], data_sharding)
        else:
            x = buf_u16[:, :-1]
            y = buf_u16[:, 1:]
        return x, y


def prepare_step_batch(bf, tokens, batch_buf, bsz, seqlen, grad_accum_steps, data_accum_sharding):
    with jax.profiler.TraceAnnotation("host_stack"):
        for micro_step in range(grad_accum_steps):
            starts, ends = bf.next_batch(bsz, seqlen)
            get_next_batch(
                starts,
                ends,
                bsz,
                seqlen,
                tokens,
                data_accum_sharding,
                batch_buf[micro_step],
                transfer_to_device=False,
            )

    with jax.profiler.TraceAnnotation("h2d"):
        stacked_batch = jnp.asarray(batch_buf, dtype=jnp.int32, device=data_accum_sharding)

    stacked_x = stacked_batch[:, :, :-1]
    stacked_y = stacked_batch[:, :, 1:]
    return stacked_x, stacked_y


def main():
    profile_start_step = int(os.environ.get("PROFILE_START_STEP", "5"))
    profile_end_step = int(os.environ.get("PROFILE_END_STEP", "15"))
    profile_logdir = Path(os.environ.get("PROFILE_LOGDIR", "profile-data")) / "chunked_cce"
    if profile_end_step <= profile_start_step:
        raise ValueError("PROFILE_END_STEP must be greater than PROFILE_START_STEP")

    devices = np.array(jax.devices())
    print("Number of devices found:", len(devices))
    data_parallel_size = tensor_parallel_size = 2
    mesh = Mesh(
        devices.reshape(data_parallel_size, tensor_parallel_size),
        axis_names=(BATCH_AXIS_NAME, TENSOR_ONLY_AXIS_NAME),
    )
    sharding_rules = ShardingRules(
        batch=BATCH_AXIS_NAME,
        vocab_out=TENSOR_ONLY_AXIS_NAME if tensor_parallel_size > 1 else None,
    )
    cfg = Config(mesh=mesh, rules=sharding_rules)

    train_files = list(Path(cfg.data_dir).glob("*train*.bin"))
    num_train_files = len(train_files)
    print("\nNumber of train files found: ", num_train_files)

    train_dl = make_grain_shard_loader(
        train_files, prefetch=16, num_threads=32, prefetch_buffer_size=16
    )
    train_iter = iter(train_dl)

    per_device_bsz = cfg.hparams.per_device_batch_size
    dp_size = cfg.mesh.shape[BATCH_AXIS_NAME]
    tp_size = cfg.mesh.shape["y"]
    bsz = per_device_bsz * dp_size
    seqlen = cfg.model.seqlen
    head_dim = cfg.model.attn.head_dim
    data_accum_sharding = logical_to_sharding(
        (None, "batch", None), cfg.mesh, cfg.rules
    )
    runtime_shardings = RuntimeShardings(
        embed_act=logical_to_sharding(
            ("batch", "sequence", "act_embed"), cfg.mesh, cfg.rules
        ),
        logits=logical_to_sharding(
            ("batch", "sequence", "vocab_out"), cfg.mesh, cfg.rules
        ),
    )

    max_lr = cfg.hparams.max_lr
    min_lr = 0.01 * max_lr
    warmup_steps = cfg.hparams.warmup_steps
    desired_batch_size = cfg.hparams.desired_batch_size
    grad_accum_steps = 2#min(4, max(2, desired_batch_size // (bsz * seqlen)))
    total_train_steps = cfg.hparams.total_train_steps

    print("Building GPT model based on the config...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    print("Model built successfully!")

    optim = optax.chain(
        optax.clip_by_global_norm(cfg.hparams.grad_clip_norm),
        build_optimizer(
            model,
            d_model=cfg.model.d_emb,
            other_peak_lr=max_lr,
            other_min_lr=min_lr,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            b1=cfg.hparams.b1,
            b2=cfg.hparams.b2,
            embedding_lr=cfg.hparams.embedding_lr,
            weight_decay=cfg.hparams.weight_decay,
            cautious_weight_decay=cfg.hparams.cautious_weight_decay,
        ),
    )

    if grad_accum_steps > 1:
        print("Using `MultiSteps` in optax for gradient accumulation...")
        optim = optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)

    optim_state = optim.init(model)

    print("")
    print("-" * 75)
    print("")
    print(line("Run name", model_run_name(cfg), value_w=30))
    print(line("Attention type", cfg.model.attn_type))
    print(line("Model dtype", str(cfg.model.dtype)))
    print(line("Num layers", cfg.model.num_layers))
    print(line("Embedding dim", cfg.model.d_emb))
    print(line("Query heads", cfg.model.q_heads))
    print(line("KV heads", cfg.model.kv_heads))
    print(line("Head dim", cfg.model.attn.head_dim))
    print(line("MLP hidden dim", cfg.model.mlp.fc1.out_features))
    print(line("Vocab size", cfg.model.vocab_size))
    print(line("Number of trainable params: ", count_params(model), comma=True))
    print(line("Sequence length per sample", seqlen))
    print(line("Per device batch size", per_device_bsz))
    print(line("Data parallel size", dp_size))
    print(line("Tensor parallel size", tp_size))
    print(line("Total batch size", bsz))
    print(line("Grad accumulation steps", grad_accum_steps))
    print()
    print(line("LR (min, max)", str((f"{min_lr:.6f}", f"{max_lr:.6f}"))))
    print(line("Warmup steps", cfg.hparams.warmup_steps))
    print(line("Weight decay", cfg.hparams.weight_decay), "\n")
    print(line("Profile loss impl", PROFILE_LOSS_IMPL))
    print(line("Profile CE chunk size", PROFILE_CE_CHUNK_SIZE))
    print(line("Profile start step", profile_start_step))
    print(line("Profile end step", profile_end_step))
    print(line("Profile logdir", str(profile_logdir)))
    print("-" * 75)

    positions = jnp.arange(seqlen)[None, :]
    with set_mesh(cfg.mesh):
        freqs = precompute_frequencies(positions=positions, features=head_dim)

    segment_ids = None
    step = 0
    num_shards_used = 0
    profiling_complete = False
    train_start_time = time.time()

    batch_tokens0 = np.empty((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)
    batch_tokens1 = np.empty((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)

    print("Starting profiling run (the first step will take some time for compilation...)\n")

    for shard in train_iter:
        if profiling_complete:
            break

        tokens = shard["tokens"]
        bos_idx = shard["bos_idx"]
        size = shard["size"]
        shard_name = Path(shard["path"]).name

        try:
            bf = BOSFinder(tokens)
            bf.bos_idx = bos_idx
            bf.size = size
            shard_processed_fully = False

            num_batches_in_shard = bf.build(bsz, seqlen)
            print(f"\n=== Processing Shard: {num_shards_used} with name: {shard_name}", end=" | ")
            print(f"Indexed {num_batches_in_shard} batches ===")

            while not shard_processed_fully and not profiling_complete:
                try:
                    if step < profile_start_step:
                        start = time.time()
                        stacked_x, stacked_y = prepare_step_batch(
                            bf,
                            tokens,
                            batch_tokens0,
                            bsz,
                            seqlen,
                            grad_accum_steps,
                            data_accum_sharding,
                        )
                        model, loss, optim_state = train_step_accum(
                            model,
                            stacked_x,
                            stacked_y,
                            segment_ids,
                            freqs,
                            optim_state,
                            optim,
                            grad_accum_steps,
                            runtime_shardings,
                        )
                        loss = jax.block_until_ready(loss)
                        end = time.time()
                        dt = end - start
                        tokens_processed = bsz * seqlen * grad_accum_steps
                        tokens_per_sec = int(tokens_processed / dt)
                        print(
                            f"Warmup step: [{step}/{profile_start_step}] | loss: {loss:8.4f} | Step time: {dt:5.2f} s | Tokens processed/s: {tokens_per_sec:>9,}"
                        )
                        step += 1
                        continue

                    profile_logdir.mkdir(parents=True, exist_ok=True)
                    jax.profiler.start_trace(str(profile_logdir))

                    stacked_x, stacked_y = prepare_step_batch(
                        bf,
                        tokens,
                        batch_tokens0,
                        bsz,
                        seqlen,
                        grad_accum_steps,
                        data_accum_sharding,
                    )

                    for profile_step in range(profile_start_step, profile_end_step):
                        with jax.profiler.StepTraceAnnotation("train", step_num=profile_step):
                            with jax.profiler.TraceAnnotation("train_step_accum"):
                                model, loss, optim_state = train_step_accum(
                                    model,
                                    stacked_x,
                                    stacked_y,
                                    segment_ids,
                                    freqs,
                                    optim_state,
                                    optim,
                                    grad_accum_steps,
                                    runtime_shardings,
                                )

                            if profile_step + 1 < profile_end_step:
                                next_x, next_y = prepare_step_batch(
                                    bf,
                                    tokens,
                                    batch_tokens1,
                                    bsz,
                                    seqlen,
                                    grad_accum_steps,
                                    data_accum_sharding,
                                )
                            else:
                                next_x, next_y = None, None

                            jax.block_until_ready(loss)

                            if next_x is not None:
                                batch_tokens0, batch_tokens1 = batch_tokens1, batch_tokens0
                                stacked_x, stacked_y = next_x, next_y

                    jax.profiler.stop_trace()
                    profiling_complete = True
                    break

                except StopIteration:
                    shard_processed_fully = True
                    num_shards_used += 1
                    print("Shard exhausted")

        finally:
            tokens.unlink_on_del()

    train_end_time = time.time()
    print(
        f"\nTotal time taken for profiling run: {(train_end_time - train_start_time) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
