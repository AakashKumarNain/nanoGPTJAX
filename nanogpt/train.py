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


import time
import warnings
import logging
from pathlib import Path
from functools import partial

import jax
import optax
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.sharding import Mesh
from jax.sharding import set_mesh

from model import count_params
from model import precompute_frequencies
from model import GPT, forward
from utils import logical_to_sharding
from fineweb_dataloader import make_grain_iter
from config import ShardingRules, Config, BATCH_AXIS_NAME

logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


def compute_loss(params, x_batch, y_batch, freqs):
    # Ensure the logits are in fp32
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


@partial(jax.jit, donate_argnums=(0,))
def val_step(params, x_batch, y_batch, freqs):
    loss, logits = compute_loss(params, x_batch, y_batch, freqs)
    return loss, logits


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
grad_accum_steps = max(4, desired_batch_size // (bsz * seqlen))
b1 = cfg.hparams.b1
b2 = cfg.hparams.b2
weight_decay = cfg.hparams.weight_decay
grad_clip_norm = cfg.hparams.grad_clip_norm
total_train_steps = cfg.hparams.total_train_steps
max_checkpoints_to_keep = cfg.ckpt_cfg.max_checkpoints_to_keep
checkpoint_save_steps = cfg.ckpt_cfg.checkpoint_save_steps


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
    optax.adamw(
        schedule, b1=b1, b2=b2, weight_decay=weight_decay, mu_dtype=jnp.float32
    ),
    # optax.contrib.muon(schedule, adam_b1=b1, adam_b2=b2, adam_weight_decay=weight_decay, weight_decay=weight_decay)
)
optim = optax.MultiSteps(base_optim, every_k_schedule=grad_accum_steps)
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

mngr = ocp.CheckpointManager(
    ckpt_path,
    handlers,
    options=options,
)


print("Building data loader...")
train_source, train_ds = make_grain_iter(
    data_dir=cfg.data_dir,
    index_path=cfg.train_idx_path,
    seqlen=seqlen,
    batch_size=bsz,
    shuffle=True,
    seed=1,
    num_threads=32,
    prefetch_buffer_size=512,
    drop_remainder=True,
)

val_source, val_ds = make_grain_iter(
    data_dir=cfg.data_dir,
    index_path=cfg.val_idx_path,
    seqlen=seqlen,
    batch_size=bsz,
    shuffle=False,
    seed=1,
    num_threads=16,
    prefetch_buffer_size=512,
    drop_remainder=True,
)
print("Data loader bult successfull!")


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
    freqs = precompute_frequencies(positions=positions, features=head_dim)


best_loss = float("inf")
patience = cfg.hparams.es_patience
patience_counter = 0
best_step = 0
last_checkpoint_step = 0

print("Starting training (the first step will take some time for compilation...)\n")
# TODO: Add validation loop after improving the dataloader

for step in range(last_checkpoint_step, total_train_steps):
    start = time.time()
    train_step_loss = 0.0
    for micro_step in range(grad_accum_steps):
        try:
            x, y = next(train_ds)
            x = jax.device_put(x, data_sharding)
            y = jax.device_put(y, data_sharding)
        except StopIteration:
            print("Data exhausted....")
            break
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
