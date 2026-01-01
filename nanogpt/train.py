import os

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
from pathlib import Path

import time
from functools import partial
import optax
import tiktoken

import jax
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.sharding import Mesh
from jax.sharding import set_mesh

from model import count_params
from model import precompute_frequencies
from model import GPT, forward
from utils import logical_to_sharding
from fineweb_dataloader import BinaryTokenDataLoader
from config import ShardingRules, Config, BATCH_AXIS_NAME

logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


def compute_loss(params, x_batch, y_batch, freqs):
    logits = forward(params, x_batch, freqs).astype(jnp.float32)
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


@partial(jax.jit, donate_argnums=(0,))
def val_step(params, x_batch, y_batch, freqs):
    loss, logits = compute_loss(params, x_batch, y_batch, freqs)
    return loss, logits


# Get the mesh, sharding rules, amd the config
devices = np.array(jax.devices())
mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
cfg = Config(mesh=mesh, rules=sharding_rules)

# Load the model
model = GPT.init(jax.random.PRNGKey(0), cfg.model)
# Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")


per_device_bsz = cfg.per_device_batch_size
# Total batch size we can fit on one device x num_devices. We are doing DDP here
bsz = per_device_bsz * len(devices)
seqlen = cfg.model.seqlen
head_dim = cfg.model.attn.head_dim
data_sharding = logical_to_sharding(("batch",), cfg.mesh, cfg.rules)

max_lr = 6e-4
min_lr = 0.0
warmup_steps = 30
desired_batch_size = 524288
grad_accum_steps = max(4, desired_batch_size // (bsz * seqlen))
b1 = 0.9
b2 = 0.95
weight_decay = 0.1
grad_clip_norm = 1.0
total_train_steps = 800
max_checkpoints_to_keep = 3
checkpoint_save_steps = 20


# LR schedule
schedule = optax.warmup_cosine_decay_schedule(
    init_value=min_lr,
    peak_value=max_lr,
    warmup_steps=warmup_steps,
    decay_steps=(total_train_steps - warmup_steps),
)

# Optimizer
base_optim = optax.chain(
    optax.clip_by_global_norm(grad_clip_norm),
    optax.adamw(schedule, b1=b1, b2=b2, weight_decay=weight_decay),
    # optax.contrib.muon(schedule, adam_b1=b1, adam_b2=b2, adam_weight_decay=weight_decay, weight_decay=weight_decay)
)
optim = optax.MultiSteps(base_optim, every_k_schedule=grad_accum_steps)
optim_state = optim.init(model)

# Checkpointing
ckpt_path = Path(cfg.ckpt_dir)
options = ocp.CheckpointManagerOptions(
    max_to_keep=max_checkpoints_to_keep, save_interval_steps=checkpoint_save_steps
)
handlers = {
    "params": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
    "optim_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
}

mngr = ocp.CheckpointManager(
    ckpt_path,
    handlers,
    options=options,
)


# Data loader
train_ds = BinaryTokenDataLoader(
    file_patterns=["fineweb10B/*train*.bin"],
    seq_len=seqlen,
    batch_size=bsz,
    bos_token=50256,
    align_to_bos=True,
    seed=123456,
    use_bos_index=True,
    num_workers=64,
    prefetch_batches=32,
)


print("")
print("-" * 75)
print("")


def line(label, value, comma=False, label_w=30, colon_w=2, value_w=20):
    fmt = f">{value_w}," if comma else f">{value_w}"
    return f"{label:<{label_w}}{':':<{colon_w}}{value:{fmt}}"


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
    freqs = precompute_frequencies(
        positions=positions, features=head_dim, dtype=cfg.model.dtype
    )


best_loss = float("inf")
patience = 30
patience_counter = 0
best_step = 0
last_checkpoint_step = 0

print("Training...\n")

for step in range(last_checkpoint_step, total_train_steps):
    start = time.time()
    train_step_loss = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_ds.get_batch()
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
    avg_val_loss = 0.0

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

    if patience_counter > patience:
        print("\nEarly stopping...")
        break

mngr.wait_until_finished()
print("Finished checkpointing! Cleaned.")
