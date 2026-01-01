import time
from functools import partial

import jax
import tiktoken
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh

from model import GPT, forward_v2
from kvcache import KVCache
from checkpoint_utils import load_weights_from_checkpoint
from config import ShardingRules, Config, BATCH_AXIS_NAME


def gather_last_logits(logits, seq_lengths):
    """Select logits at the final valid position for each sequence."""
    seq_len = logits.shape[1]
    indices = jnp.clip(seq_lengths - 1, 0, seq_len - 1)
    return jax.vmap(lambda logit, idx: logit[idx])(logits, indices)


@partial(jax.jit, static_argnames=("pad_id", "head_dim"))
def prefill_step(params, tokens, cache, pad_id, head_dim):
    """Run a prefill pass and return next-token logits plus updated cache."""
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    segment_ids = (tokens != pad_id).astype(jnp.int32)
    logits, cache = forward_v2(params, tokens, segment_ids, cache, head_dim)
    seq_lengths = jnp.sum(segment_ids != 0, axis=-1).astype(jnp.int32)
    next_logits = gather_last_logits(logits, seq_lengths)
    return next_logits, cache


def decode_step(params, tokens, cache, head_dim):
    """Run a single-token decode step and return logits plus updated cache."""
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    segment_ids = jnp.ones_like(tokens, dtype=jnp.int32)
    logits, cache = forward_v2(params, tokens, segment_ids, cache, head_dim)
    return logits[:, -1, :], cache


def sample_from_logits(logits, rng, temperature=1.0, top_k=0):
    logits = logits.astype(jnp.float32)
    vocab = logits.shape[-1]

    # Greedy path: temperature <= 0
    if temperature <= 0.0:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    # Top-k filtering (order vs temperature is immaterial for top-k)
    if top_k is not None:
        k = int(top_k)
        if 0 < k < vocab:
            sorted_logits = jnp.sort(logits, axis=-1)
            threshold = sorted_logits[:, -k][:, None]  # kth largest logit
            logits = jnp.where(logits < threshold, -jnp.inf, logits)

    # Temperature scaling
    tiny = jnp.finfo(jnp.float32).tiny
    logits = logits / jnp.maximum(temperature, tiny)

    # Sample
    return jax.random.categorical(rng, logits, axis=-1).astype(jnp.int32)


@partial(
    jax.jit, static_argnames=("temperature", "head_dim", "max_new_tokens", "top_k")
)
def decode(
    params,
    cache,
    head_dim,
    last_token,
    decode_key,
    generated,
    temperature,
    top_k,
    max_new_tokens,
):
    def decode_body(carry, t):
        cache, last_token, decode_key, generated = carry
        logits, cache = decode_step(params, last_token, cache, head_dim)
        decode_key, sub = jax.random.split(decode_key)
        token = sample_from_logits(logits.astype(jnp.float32), sub, temperature, top_k)
        generated = generated.at[:, t].set(token)
        return (cache, token[:, None], decode_key, generated), None

    (cache, last_token, decode_key, generated), _ = jax.lax.scan(
        decode_body,
        (cache, last_token, decode_key, generated),
        jnp.arange(1, max_new_tokens),
    )
    return generated


def pad_prompts(prompts, tokenizer, pad_id, round_to_power_of_two=True):
    if isinstance(prompts, str):
        prompts = [prompts]
    encoded = tokenizer.encode_batch(prompts, allowed_special={"<|endoftext|>"})
    if not encoded:
        raise ValueError("No prompts provided.")
    max_len = max(len(seq) for seq in encoded)
    if max_len == 0:
        max_len = 1
    if round_to_power_of_two:
        pad_len = 1 << (max_len - 1).bit_length()
    else:
        pad_len = max_len
    batch, seq_lengths = [], []
    for seq in encoded:
        if not seq:
            seq = [pad_id]
        seq_lengths.append(len(seq))
        pad_width = pad_len - len(seq)
        batch.append(seq + [pad_id] * pad_width)
    tokens = jnp.array(batch, dtype=jnp.int32)
    seq_lengths = jnp.array(seq_lengths, dtype=jnp.int32)
    return tokens, seq_lengths


def sample_from_model(
    key,
    prompts,
    tokenizer,
    params,
    max_new_tokens,
    pad_id,
    head_dim,
    temperature=1.0,
    top_k=500,
):
    # TODO: Move it out of this functions and shard the tokens
    tokens, _ = pad_prompts(
        prompts, tokenizer, pad_id=pad_id, round_to_power_of_two=True
    )
    batch_size = tokens.shape[0]

    cache_key, decode_key = jax.random.split(key)

    # Cache init
    cache = KVCache.init(cache_key, cfg.mesh, cfg.rules, batch_size, cfg)

    # Prefill
    logits, cache = prefill_step(params, tokens, cache, pad_id, head_dim)

    decode_key, sub = jax.random.split(decode_key)
    next_tokens = jax.jit(sample_from_logits, static_argnames=("temperature", "top_k"))(
        logits.astype(jnp.float32), sub, temperature, top_k
    )

    generated = (
        jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
        .at[:, 0]
        .set(next_tokens)
    )
    last_token = next_tokens[:, None]

    # Decode loop via jitted scan
    if max_new_tokens > 1:
        generated = decode(
            params,
            cache,
            head_dim,
            last_token,
            decode_key,
            generated,
            temperature,
            top_k,
            max_new_tokens,
        )

    return generated


if __name__ == "__main__":
    devices = np.array(jax.devices())
    mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
    sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
    cfg = Config(mesh=mesh, rules=sharding_rules)

    # Get the weight shardings
    model_sharding = GPT.shardings(cfg.mesh, cfg.rules, cfg.model)
    model = load_weights_from_checkpoint(cfg.load_ckpt_path, model_sharding)
    print("Model weights loaded from the checkpoint successfully!\n")

    tokenizer = tiktoken.get_encoding("gpt2")
    PAD_ID = tokenizer.eot_token
    max_new_tokens = 100
    top_k = 500

    jax.set_mesh(cfg.mesh)

    # Warmup
    prompts = ["Did you hear the noise coming"] * 4
    key = jax.random.PRNGKey(0)
    print("Warming up the model...")

    for _ in range(3):
        key, subkey = jax.random.split(key)
        _ = sample_from_model(
            subkey,
            prompts,
            tokenizer,
            model,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            pad_id=PAD_ID,
            head_dim=cfg.model.attn.head_dim,
        )
    print("Warming up complete!\nGenerating...")

    prompts = [
        "Did you notice that this world",
        "Hear that?",
        "Hello World! My dear",
        "Some say we are tired far",
    ]
    key, subkey = jax.random.split(key)
    start = time.perf_counter()
    out = sample_from_model(
        subkey,
        prompts,
        tokenizer,
        model,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        pad_id=PAD_ID,
        head_dim=cfg.model.attn.head_dim,
    )
    end = time.perf_counter()
    print(
        f"Time taken to generate {max_new_tokens * len(prompts)} tokens: {(end - start) * 1000:.2f} ms\n"
    )
    decoded = tokenizer.decode_batch(out.tolist())

    for p, d in zip(prompts, decoded):
        print(f"Prompt: {p}\n")
        print(f"Completion: {d}")
        print("-" * 75)
