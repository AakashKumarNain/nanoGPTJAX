import time
import math
import dataclasses
from functools import partial

import jax
import tiktoken
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh

from model import GPT, forward_v2
from kvcache import KVCache, count_left_padding, prepare_chunk
from checkpoint_utils import load_weights_from_checkpoint
from config import ShardingRules, Config, BATCH_AXIS_NAME


@partial(jax.jit, static_argnames=("head_dim", "pad_id"))
def prefill(params, input_ids, segment_ids, cache, head_dim, pad_id):
    """Prefill with LEFT-padded sequences.

    With left padding, starts[b] = count of left pad tokens for sequence b.
    After prefill, cache.iter = seq_len % cache.size (same for all sequences due to alignment).
    """

    left_pad_counts = count_left_padding(input_ids, pad_id=pad_id)
    uninitialized_iter = -jnp.ones_like(cache.iter)
    cache = dataclasses.replace(cache, starts=left_pad_counts, iter=uninitialized_iter)
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    # With left padding, last valid token is always at position -1 (rightmost)
    last_token_logits = logits[:, -1, :]
    next_tokens = jnp.argmax(last_token_logits, axis=-1)
    return logits, next_tokens, cache


def decode(params, input_ids, cache, head_dim):
    """Decode step for LEFT-padded sequences.

    All sequences generate at the same position since they're aligned at the right.
    """

    segment_ids = jnp.ones_like(input_ids, dtype=jnp.int32)
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    return logits[:, -1, :], cache


def sample_from_logits(logits, rng, temperature=1.0, top_k=0):
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


@partial(jax.jit, static_argnames=("temperature", "head_dim", "max_new_tokens", "top_k"))  # fmt: off
def generate(
    params,
    cache,
    last_token,
    generated_tokens,
    head_dim,
    decode_key,
    temperature,
    top_k,
    max_new_tokens,
):
    def decode_body(carry, t):
        cache, last_token, decode_key, generated_tokens = carry
        logits, cache = decode(params, last_token, cache, head_dim)
        decode_key, sub = jax.random.split(decode_key)
        token = sample_from_logits(logits, sub, temperature, top_k)
        generated_tokens = generated_tokens.at[:, t].set(token)
        return (cache, token[:, None], decode_key, generated_tokens), None

    (cache, last_token, decode_key, generated_tokens), _ = jax.lax.scan(
        decode_body,
        (cache, last_token, decode_key, generated_tokens),
        jnp.arange(1, max_new_tokens),
    )
    return generated_tokens


def pad_tokens(tokens, pad_id, pad_to_power_of_two=False):
    curr_max_len = max([len(s) for s in tokens])
    if pad_to_power_of_two:
        pad_to = 2 ** math.ceil(math.log2((curr_max_len)))
    else:
        pad_to = curr_max_len
    padded = []
    segment_ids = []

    for encoded in tokens:
        p, s = prepare_chunk(jnp.array(encoded), pad_id=pad_id, pad_to=pad_to)
        padded.append(p[0])
        segment_ids.append(s[0])

    padded = jnp.stack(padded)
    segment_ids = jnp.stack(segment_ids)
    return padded, segment_ids


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
    pad_to_power_of_two=True,
):
    # TODO: Move it out of this functions and shard the tokens
    encoded = tokenizer.encode_batch(
        prompts, allowed_special={"<|endoftext|>", "<|pad|>"}
    )
    input_ids, segment_ids = pad_tokens(
        encoded, pad_to_power_of_two=pad_to_power_of_two
    )

    cache_key, decode_key = jax.random.split(key)

    batch_size = input_ids.shape[0]
    cache = KVCache.init(cache_key, cfg.mesh, cfg.rules, batch_size, cfg)
    logits, next_tokens, cache = prefill(
        model, input_ids, segment_ids, cache, head_dim, pad_id=PAD_ID
    )
    generated_tokens = (
        jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
        .at[:, 0]
        .set(next_tokens)
    )

    # Decode loop via jitted scan
    if max_new_tokens > 1:
        last_token = next_tokens[:, None]
        top_k = 0 if top_k is None else int(top_k)
        generated = generate(
            model,
            cache,
            last_token,
            generated_tokens,
            head_dim,
            decode_key,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )

    return generated


if __name__ == "__main__":
    devices = np.array(jax.devices())
    print("Found devices: ", devices)
    print("Platform: ", devices[0].platform)
    mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
    sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
    cfg = Config(mesh=mesh, rules=sharding_rules)

    # Get the weight shardings
    model_sharding = GPT.shardings(cfg.mesh, cfg.rules, cfg.model)
    print("Loading weights from the checkpoint...")
    model = load_weights_from_checkpoint(cfg.load_ckpt_path, model_sharding)
    print("Weights loaded from the checkpoint successfully!")

    base_tokenizer = tiktoken.get_encoding("gpt2")
    PAD_TOKEN = "<|pad|>"

    tokenizer = tiktoken.Encoding(
        name="gpt2_with_pad",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens={
            **base_tokenizer._special_tokens,
            PAD_TOKEN: base_tokenizer.n_vocab,  # next available token id
        },
    )

    PAD_ID = tokenizer.encode(PAD_TOKEN, allowed_special={"<|pad|>"})[0]
    max_new_tokens = 100
    top_k = None

    jax.set_mesh(cfg.mesh)

    # Warmup
    prompts = ["<|endoftext|>Did you hear the noise coming "] * 4
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
        "<|endoftext|>Did you notice that this world",
        "<|endoftext|>Hear that?",
        "<|endoftext|>Hello World! My dear",
        "<|endoftext|>Some say we are tired far",
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
        print(f"Prompt: {p}")
        print(f"Completion: {d}")
        print("-" * 75)
