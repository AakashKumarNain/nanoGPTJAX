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
    "--xla_gpu_enable_while_loop_double_buffering=true "
    "--xla_gpu_enable_pipelined_p2p=true "
    "--xla_gpu_collective_permute_decomposer_threshold=1024 "
)
import warnings
import logging
import time
from pathlib import Path
from functools import partial

import jax

jax.config.update("jax_optimization_level", "O1")

import optax
import grain
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.sharding import Mesh

from model import count_params
from model import precompute_frequencies
from model import GPT, forward
from utils import logical_to_sharding
from checkpoint_utils import load_weights_from_checkpoint_with_validation
from config import ShardingRules, Config, BATCH_AXIS_NAME
from sft_dataloader import make_grain_shard_loader, build_tokenizer


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


# For RoPE frequency calculations
jitted_precompute_frequencies = jax.jit(
    precompute_frequencies, static_argnames=("features", "theta", "dtype")
)


def compute_loss(params, x_batch, y_batch, segment_ids, freqs, loss_mask):
    logits = forward(params, x_batch, segment_ids, freqs)
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


@partial(
    jax.jit,
    static_argnames=("optim", "grad_accum_steps"),
    donate_argnums=(0, 1, 3, 4, 5, 7),
)
def train_step_accum(
    params,
    x_batch,
    y_batch,
    segment_ids,
    freqs,
    optim_state,
    optim,
    loss_mask,
    grad_accum_steps,
):
    def body(carry, xy):
        param, opt_state, lsum = carry
        xb, yb = xy
        loss, grad = jax.value_and_grad(compute_loss)(
            param, xb, yb, segment_ids, freqs, loss_mask
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


@partial(jax.jit, static_argnames=("optim",), donate_argnums=(0, 1, 3, 4, 5))
def train_step(
    params, x_batch, y_batch, segment_ids, freqs, optim_state, optim, loss_mask
):
    loss, grads = jax.value_and_grad(compute_loss)(
        params, x_batch, y_batch, segment_ids, freqs, loss_mask
    )
    updates, optim_state = optim.update(grads, optim_state, params)
    updated_params = optax.apply_updates(params, updates)
    return updated_params, loss, optim_state


@jax.jit
def val_step(params, x_batch, y_batch, segment_ids, freqs, loss_mask):
    loss = compute_loss(params, x_batch, y_batch, segment_ids, freqs, loss_mask)
    return loss


def line(label, value, comma=False, label_w=30, colon_w=2, value_w=20):
    fmt = f">{value_w}," if comma else f">{value_w}"
    return f"{label:<{label_w}}{':':<{colon_w}}{value:{fmt}}"


def main():
    # Get the mesh, sharding rules, amd the config
    devices = np.array(jax.devices())
    print("Number of devices found:", len(devices))
    mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
    sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
    cfg = Config(mesh=mesh, rules=sharding_rules)

    per_device_bsz = cfg.hparams.per_device_batch_size
    bsz = per_device_bsz * len(devices)
    seqlen = cfg.model.seqlen
    head_dim = cfg.model.attn.head_dim
    data_sharding = logical_to_sharding(("batch",), cfg.mesh, cfg.rules)
    data_accum_sharding = logical_to_sharding(
        (None, "batch", None), cfg.mesh, cfg.rules
    )

    max_lr = cfg.hparams.max_lr
    min_lr = 0.01 * max_lr
    warmup_steps = cfg.hparams.warmup_steps
    desired_batch_size = cfg.hparams.desired_batch_size
    grad_accum_steps = 1

    total_train_steps = cfg.hparams.total_train_steps
    max_checkpoints_to_keep = cfg.ckpt_cfg.max_checkpoints_to_keep
    checkpoint_save_steps = cfg.ckpt_cfg.checkpoint_save_steps

    tok_info = build_tokenizer()
    train_dl = make_grain_shard_loader(
        data_dir=cfg.data_dir,
        split="train",
        pad_id=tok_info["pad_id"],
        batch_size=bsz,
        sequence_length=seqlen,
        grad_accum_steps=1,
        data_sharding=data_sharding,
        multi_threading=True,
    )
    train_iter = iter(train_dl)

    # During testing, we had only one shard of validation data.
    # Hence multi-threading was turned off for it.
    # TODO: Enable multi-threading once we have enough val shards
    val_dl = make_grain_shard_loader(
        data_dir=cfg.data_dir,
        split="test",
        pad_id=tok_info["pad_id"],
        batch_size=bsz,
        sequence_length=seqlen,
        grad_accum_steps=1,
        data_sharding=data_sharding,
        cycle_length=1,
        multi_threading=False,
    )

    # Load the model
    print("Building GPT model based on the config...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    print("Model built successfully!")
    model_sharding = GPT.shardings(cfg.mesh, cfg.rules, cfg.model)
    model = load_weights_from_checkpoint_with_validation(
        cfg.ckpt_cfg.load_params_ckpt_path, model, model_sharding
    )
    print("Weights loaded from the checkpoint successfully!")

    # Optimizer
    optim = optax.chain(
        optax.clip_by_global_norm(cfg.hparams.grad_clip_norm),
        # TODO: Replace it. This was used just for testing
        optax.adamw(learning_rate=1e-4),
    )
    optim_state = optim.init(model)

    #  Checkpointing
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints_to_keep,
        save_interval_steps=checkpoint_save_steps,
        enable_async_checkpointing=True,
        enable_background_delete=True,
    )
    handlers = {
        "params": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "optim_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        # "ds": ocp.Checkpointer(grain.checkpoint.CheckpointHandler()),
    }
    mngr = ocp.CheckpointManager(cfg.ckpt_cfg.save_ckpt_dir, handlers, options=options)

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
    print(line("Warmup steps", cfg.hparams.warmup_steps))
    print(line("Weight decay", cfg.hparams.weight_decay), "\n")
    print("-" * 75)

    best_loss = float("inf")
    last_val_loss = float("inf")
    es_patience = cfg.hparams.es_patience
    es_patience_counter = 0
    best_step = 0
    num_shards_used = 0
    total_tokens_consumed = 0

    step = cfg.ckpt_cfg.resume_from_step
    print("Starting training (the first step will take some time for compilation...)\n")

    training_complete = False
    train_start_time = time.time()

    for train_batch in train_iter:
        if training_complete:
            break
        start = time.time()
        x, y = train_batch["x"], train_batch["y"]
        segment_ids = train_batch["segment_ids"]
        completion_mask = train_batch["completion_mask"]
        positions = train_batch["positions"]

        with jax.set_mesh(cfg.mesh):
            freqs = jitted_precompute_frequencies(positions, head_dim)

        model, loss, optim_state = train_step(
            model, x, y, segment_ids, freqs, optim_state, optim, completion_mask
        )
        loss = jax.block_until_ready(loss).item()
        end = time.time()
        dt = end - start
        train_time_elapsed = (end - train_start_time) / 60  # in minutes

        step += 1
        tokens_processed = bsz * seqlen * grad_accum_steps
        total_tokens_consumed += tokens_processed
        tokens_per_sec = int(tokens_processed / dt)

        print(
            f"Step: [{str(step).zfill(len(str(total_train_steps)))}/{total_train_steps}] | loss: {loss:8.4f} | Step time: {dt:5.2f} s | Train time: {train_time_elapsed:6.2f} min | Tokens processed/s: {tokens_per_sec:>9,}"
        )

        if (step % options.save_interval_steps) == 0:
            mngr.save(
                step,
                args=ocp.args.Composite(
                    params=ocp.args.PyTreeSave(model),
                    optim_state=ocp.args.PyTreeSave(optim_state),
                    # ds=grain.checkpoint.CheckpointSave(train_iter),
                ),
            )
            print("\nScoring model performance on validation data...\n")
            val_loss = 0.0
            val_steps_count = 0
            val_iter = iter(val_dl)
            for val_batch in val_iter:
                val_x, val_y = val_batch["x"], val_batch["y"]
                val_segment_ids = val_batch["segment_ids"]
                val_completion_mask = val_batch["completion_mask"]
                val_positions = val_batch["positions"]
                with jax.set_mesh(cfg.mesh):
                    val_freqs = jitted_precompute_frequencies(val_positions, head_dim)
                loss = val_step(
                    model, val_x, val_y, val_segment_ids, val_freqs, val_completion_mask
                )

                val_loss += loss
                val_steps_count += 1
            avg_val_loss = val_loss / val_steps_count
            avg_val_loss = jax.block_until_ready(avg_val_loss).item()

            improved = avg_val_loss < best_loss
            if improved:
                best_loss = avg_val_loss
                best_step = step
                es_patience_counter = 0
            else:
                es_patience_counter += 1

            if es_patience_counter > es_patience:
                print(
                    f"\nEarly stopping triggered! No improvement for {es_patience_counter} steps."
                )
                print(f"Total number of shards consumed : {num_shards_used}")
                print(
                    f"Best loss                       : {best_loss:.4f} at step {best_step}"
                )
                mngr.wait_until_finished()
                training_complete = True
                break

            print(f"last_val_loss : {last_val_loss:.4f}")
            print(f"curr_val_loss : {avg_val_loss:.4f}")
            print(f"Best loss     : {best_loss:.4f} at step {best_step}\n")
            last_val_loss = avg_val_loss

        if step >= total_train_steps:
            print(f"\nReached maximum training steps  : {total_train_steps}")
            print(f"Total number of shards consumed : {num_shards_used}")
            print(f"Best loss : {best_loss:.4f} at step {best_step}")
            mngr.wait_until_finished()
            print("Finished checkpointing! Cleaned.")
            # training_complete = True
            break

    train_end_time = time.time()
    print(
        f"\nTotal time taken to train the model: {(train_end_time - train_start_time) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
