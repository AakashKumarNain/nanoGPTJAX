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

from sft_dataloader import make_grain_shard_loader

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
from optim import build_optimizer
from utils import logical_to_sharding
from checkpoint_utils import load_weights_from_checkpoint
from config import ShardingRules, BATCH_AXIS_NAME
from config import Config, HyperParams, CheckpointConfig


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


# TODO: We should not JIT it here, and leave the JIT compilation in the outer
# function. We need to fix this, but later!
jitted_precompute_frequencies = jax.jit(
    precompute_frequencies, static_argnames=("features", "theta", "dtype")
)


def compute_loss(params, x_batch, y_batch, segment_ids, freqs, loss_mask):
    """Corss-entropy loss with masked tokens"""
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


@partial(jax.jit, static_argnames=("optim", "head_dim"), donate_argnums=(0, 1, 5))
def train_step(
    params,
    x_batch,
    y_batch,
    segment_ids,
    positions,
    optim_state,
    optim,
    head_dim,
    prompt_mask,
):
    freqs = jitted_precompute_frequencies(positions, head_dim)
    loss, grads = jax.value_and_grad(compute_loss)(
        params, x_batch, y_batch, segment_ids, freqs, prompt_mask
    )
    updates, optim_state = optim.update(grads, optim_state, params)
    updated_params = optax.apply_updates(params, updates)
    return updated_params, loss, optim_state


# fmt: off
@partial(
    jax.jit,
    static_argnames=("optim", "grad_accum_steps", "head_dim"),
    donate_argnums=(0, 1, 3, 4, 5,),
)
def train_step_accum(
    params,
    x_batch, y_batch,  
    segment_ids, positions,  
    optim_state, optim,
    head_dim,
    prompt_mask,
    grad_accum_steps,
):
# fmt: on
    freqs = jax.vmap(jitted_precompute_frequencies, in_axes=(0, None))(
        positions, head_dim
    )

    def body(carry, microbatch):
        p, gsum, lsum = carry

        xb, yb, segb, freqb, maskb = microbatch
        loss, g = jax.value_and_grad(compute_loss)(p, xb, yb, segb, freqb, maskb)

        gsum = jax.tree_util.tree_map(lambda a, b: a + b, gsum, g)
        lsum = lsum + loss
        return (p, gsum, lsum), None

    g0 = jax.tree_util.tree_map(jnp.zeros_like, params)
    carry0 = (params, g0, jnp.array(0.0, dtype=jnp.result_type(0.0)))
    (p, gsum, lsum), _ = jax.lax.scan(
        body,
        carry0,
        (x_batch, y_batch, segment_ids, freqs, prompt_mask),
        length=grad_accum_steps,
    )

    steps = grad_accum_steps
    gsum = jax.tree_util.tree_map(lambda g: g / steps, gsum)
    loss = lsum / steps

    updates, optim_state = optim.update(gsum, optim_state, p)
    params = optax.apply_updates(p, updates)
    return params, loss, optim_state


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
    hparams = HyperParams(
        b1=0.8,
        b2=0.95,
        other_peak_lr=0.02,
        total_train_steps=1000,
        embedding_lr=0.2,
        unembedding_lr=0.004,
        weight_decay=0.0,
        cautious_weight_decay=0.0,
        init_lr_frac=0.2,
        final_lr_frac=0,
        es_patience=50,
    )
    # TODO: Remove this from here and instantiate directly in the config
    # This is just for experimentation for now
    ckpt_cfg = CheckpointConfig(
        save_ckpt_dir="/home/ubuntu/nanochat/jaxnano/ckpts/sft/exp1/",
        load_params_ckpt_path="/home/ubuntu/nanochat/jaxnano/ckpts/optim_exp_muon/exp6/5700/params",
    )

    cfg = Config(
        mesh=mesh,
        rules=sharding_rules,
        hparams=hparams,
        ckpt_cfg=ckpt_cfg,
        data_dir="/home/ubuntu/nanochat/jaxnano/sft_data/",
    )

    per_device_bsz = cfg.hparams.per_device_batch_size
    bsz = per_device_bsz * len(devices)
    seqlen = cfg.model.seqlen
    head_dim = cfg.model.attn.head_dim
    data_sharding = logical_to_sharding(("batch",), cfg.mesh, cfg.rules)
    data_accum_sharding = logical_to_sharding(
        (None, "batch", None), cfg.mesh, cfg.rules
    )

    desired_batch_size = cfg.hparams.desired_batch_size
    grad_accum_steps = max(2, desired_batch_size // (bsz * seqlen))

    total_train_steps = cfg.hparams.total_train_steps
    max_checkpoints_to_keep = cfg.ckpt_cfg.max_checkpoints_to_keep
    checkpoint_save_steps = cfg.ckpt_cfg.checkpoint_save_steps

    train_dl = make_grain_shard_loader(
        batch_size=bsz,
        sequence_length=seqlen + 1,
        grad_accum_steps=grad_accum_steps,  # gradient accumulation for training
        data_sharding=data_accum_sharding if grad_accum_steps > 1 else data_sharding,
        data_dir=cfg.data_dir,
        split="train",
        repeat=False,  # repeat for multiple epochs if you have less data
    )
    val_dl = make_grain_shard_loader(
        batch_size=bsz,
        sequence_length=seqlen + 1,
        grad_accum_steps=0,  # no accum steps for validation step
        data_sharding=data_sharding,
        data_dir=cfg.data_dir,
        split="test",
        repeat=False,  # just load once
    )
    train_iter = iter(train_dl)

    # Load the model from the pretraining checkpoint
    model_sharding = GPT.shardings(cfg.mesh, cfg.rules, cfg.model)
    model = load_weights_from_checkpoint(
        cfg.ckpt_cfg.load_params_ckpt_path, model_sharding
    )
    print("Weights loaded from the checkpoint successfully!")

    # Optimizer
    optim = optax.chain(
        optax.clip_by_global_norm(cfg.hparams.grad_clip_norm),
        build_optimizer(
            model,
            d_model=cfg.model.d_emb,
            other_peak_lr=cfg.hparams.other_peak_lr,
            embedding_lr=cfg.hparams.embedding_lr,
            total_train_steps=total_train_steps,
            cautious_weight_decay=0.0,
        ),
    )

    optim_state = optim.init(model)

    ckpt_path = Path(cfg.ckpt_cfg.save_ckpt_dir)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints_to_keep,
        save_interval_steps=checkpoint_save_steps,
        enable_async_checkpointing=True,
        enable_background_delete=True,
    )

    handlers = {
        "params": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "optim_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "ds": ocp.Checkpointer(grain.checkpoint.CheckpointHandler()),
    }

    mngr = ocp.CheckpointManager(ckpt_path, handlers, options=options)

    print("")
    print("-" * 75)
    print("")

    print(line("Number of trainable params: ", count_params(model), comma=True))
    print(line("Sequence length per sample", seqlen))
    print(line("Per device batch size", per_device_bsz))
    print(line("Total batch size", bsz))
    print(line("Grad accumulation steps", grad_accum_steps))
    print(line("Weight decay", cfg.hparams.weight_decay), "\n")
    print("-" * 75)

    resume_from_step = cfg.ckpt_cfg.last_checkpoint_step

    if resume_from_step > 0:
        # resume_ckpt_path = os.path.join(cfg.ckpt_cfg.save_ckpt_dir, str(resume_from_step))
        resume_ckpt_path = os.path.join(
            cfg.ckpt_cfg.save_ckpt_dir, str(resume_from_step)
        )
        if os.path.exists(resume_ckpt_path):
            from checkpoint_utils import load_checkpoint

            model, optim_state = load_checkpoint(
                mngr, resume_from_step, model, optim_state, mesh, ds_iter=None
            )
        else:
            resume_from_step = 0
            print(
                f"Checkpoint path {resume_ckpt_path} not found! Resuming training without restoring checkpoint..."
            )

    best_loss = float("inf")
    last_val_loss = float("inf")
    es_patience = cfg.hparams.es_patience
    es_patience_counter = 0
    best_step = 0
    num_shards_used = 0
    total_tokens_consumed = 0

    step = resume_from_step
    print("Starting training (the first step will take some time for compilation...)\n")

    training_complete = False
    train_start_time = time.time()

    # Training loop with explicit counter
    while not training_complete:
        try:
            start = time.time()
            batch = next(train_iter)
            if grad_accum_steps > 1:
                model, loss, optim_state = train_step_accum(
                    model,
                    batch["x"],
                    batch["y"],
                    batch["segment_ids"],
                    batch["positions"],
                    optim_state,
                    optim,
                    head_dim,
                    batch["completion_mask"],
                    grad_accum_steps,
                )
            else:
                model, loss, optim_state = train_step(
                    model,
                    batch["x"],
                    batch["y"],
                    batch["segment_ids"],
                    batch["positions"],
                    optim_state,
                    optim,
                    head_dim,
                    batch["completion_mask"],
                )
            jax.block_until_ready(loss)

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
                        ds=grain.checkpoint.CheckpointSave(train_iter),
                    ),
                )

            if step % cfg.hparams.val_interval == 0:
                print("\nScoring model performance on vdalidation data...\n")
                val_loss = 0.0
                val_steps_count = 0
                val_iter = iter(val_dl)
                while True:
                    try:
                        batch = next(val_iter)
                        freqs = jitted_precompute_frequencies(
                            batch["positions"], head_dim
                        )
                        loss = val_step(
                            model,
                            batch["x"],
                            batch["y"],
                            batch["segment_ids"],
                            freqs,
                            batch["completion_mask"],
                        )
                        val_loss += loss
                        val_steps_count += 1
                    except StopIteration:
                        break
                avg_val_loss = val_loss / val_steps_count
                jax.block_until_ready(avg_val_loss)

                improved = avg_val_loss < best_loss
                if improved:
                    best_loss = avg_val_loss
                    best_step = step
                    es_patience_counter = 0
                else:
                    es_patience_counter += 1

                 # fmt: off
                if es_patience_counter > es_patience:
                    print(f"\nEarly stopping triggered! No improvement for {es_patience_counter} steps.")
                    print(f"Total number of shards consumed : {num_shards_used}")
                    print(f"Best loss                       : {best_loss:.4f} at step {best_step}")
                    mngr.wait_until_finished()
                    training_complete = True
                    break
                 # fmt: on

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
                training_complete = True
                break

        except StopIteration:
            training_complete = True
            mngr.wait_until_finished()
            print("Finished checkpointing! Cleaned.")

    train_end_time = time.time()
    print(
        f"\nTotal time taken to train the model: {(train_end_time - train_start_time) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()
