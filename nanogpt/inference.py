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
from checkpoint_utils import load_weights_from_checkpoint_with_validation
from config import ShardingRules, Config, BATCH_AXIS_NAME

from sft_dataloader import build_tokenizer
from sft_dataloader import format_conversation


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
    # Ideal order for most use cases is: temperature scaling -> top_k -> top_p
    vocab = logits.shape[-1]

    # Greedy path: temperature <= 0
    if temperature <= 0.0:
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)
    else:
        # Temperature scaling
        tiny = jnp.finfo(jnp.float32).tiny
        logits = logits / jnp.maximum(temperature, tiny)

    if top_k is not None:
        k = int(top_k)
        if 0 < k < vocab:

            def set_values(arr, idx, val):
                return arr.at[idx].set(val)

            values, indices = jax.lax.top_k(logits, k)
            filtered_logits = jnp.full_like(logits, fill_value=-jnp.inf)
            logits = jax.vmap(set_values, in_axes=(0, 0, 0))(
                filtered_logits, indices, values
            )
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


# This version can be used when you want to put a certain stop condition (e.g. max_new_tokens)
# or stop_token in the loop and break out of it ASAP. scan does not let you break out of the
# loop, but while_loop does. Though it allows you to break out early, you will still be using
# same amount of memory as in scan!
@partial(jax.jit, static_argnames=("head_dim", "max_new_tokens", "temperature", "top_k", "stop_token_id"))  # fmt: off
def generate_v2(
    params,
    cache,
    last_token,
    generated_tokens,
    head_dim,
    decode_key,
    temperature,
    top_k,
    max_new_tokens,
    stop_token_id,
):
    def cond_fn(state):
        _, _, _, generated_tokens, step, finished = state
        return (~finished) & (step < max_new_tokens)

    def body_fn(state):
        cache, last_token, decode_key, generated_tokens, step, finished = state
        logits, cache = decode(params, last_token, cache, head_dim)
        decode_key, sub = jax.random.split(decode_key)
        token = sample_from_logits(logits, sub, temperature, top_k)
        generated_tokens = generated_tokens.at[:, step].set(token)
        finished = jnp.all(token == stop_token_id)
        return (cache, token[:, None], decode_key, generated_tokens, step + 1, finished)

    state = (
        cache,
        last_token,
        decode_key,
        generated_tokens,
        jnp.int32(1),
        jnp.bool_(False),
    )
    _, _, _, generated_tokens, _, _ = jax.lax.while_loop(cond_fn, body_fn, state)
    return generated_tokens


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


if __name__ == "__main__":
    devices = np.array(jax.devices())
    print("Found devices: ", devices)
    print("Platform: ", devices[0].platform)
    mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
    sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
    cfg = Config(mesh=mesh, rules=sharding_rules)

    # Although the model structure does not change between pretraining and SFT
    # the performance of the model depends on the checkpoint being used. If you
    # are testing for SFT, ensure you load the right checkpoint and test the model
    # in the right way. A big difference between pretraining and SFT model is that
    # the SFTed model would expect a chat_templated input(`format_conversation`
    # function output in our case)
    MODEL_TYPE = "pretrained"

    # Get the weight shardings
    print("Building GPT model based on the config...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    model_sharding = GPT.shardings(cfg.mesh, cfg.rules, cfg.model)
    print("Model built successfully!\n")
    model = load_weights_from_checkpoint_with_validation(
        cfg.ckpt_cfg.load_params_ckpt_path, model, model_sharding
    )
    print("Weights loaded from the checkpoint successfully!")

    tok_info = build_tokenizer()
    tokenizer = tok_info["tokenizer"]
    user_start = tok_info["user_start"]
    user_end = tok_info["user_end"]
    assistant_start = tok_info["assistant_start"]
    assistant_end = tok_info["assistant_end"]
    pad_token = tok_info["pad_token"]
    # custom_tokens = [user_start, user_end, assistant_start, assistant_end, pad_token]
    custom_tokens = [t for t in tok_info["custom_token_ids"]]

    BOS_ID = tok_info["bos_id"]
    PAD_ID = tok_info["pad_id"]
    BOS = tok_info["bos"]

    head_dim = cfg.model.attn.head_dim
    max_new_tokens = 100
    top_k = 500
    temperature = 0.8

    if MODEL_TYPE == "pretrained":
        # Warmup the model with some fake inputs first
        prompts = ["<|endoftext|>Did you hear the noise coming "] * len(devices)
        encoded = tokenizer.encode_batch(prompts, allowed_special="all")
        input_ids, segment_ids = pad_tokens(encoded, pad_to_power_of_two=True)
        key = jax.random.PRNGKey(123)
        print("Warming up the model...")

        # Set the cache
        cache_key = jax.random.PRNGKey(1)
        batch_size = input_ids.shape[0]
        cache = KVCache.init(cache_key, cfg.mesh, cfg.rules, batch_size, cfg)

        # Always make the compiler aware of the mesh context
        # so that it does not do stupid things by trying being
        # more smart!
        with jax.set_mesh(cfg.mesh):
            key, subkey = jax.random.split(key)
            _, next_tokens, temp_cache = prefill(
                model, input_ids, segment_ids, cache, head_dim, pad_id=PAD_ID
            )

            generated_tokens = (
                jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
                .at[:, 0]
                .set(next_tokens)
            )
            last_token = next_tokens[:, None]
            generated = generate(
                model,
                temp_cache,
                last_token,
                generated_tokens,
                head_dim,
                subkey,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )
        print("Warming up complete!\nGenerating...")

        prompts = [
            "<|endoftext|>Did you notice that this world",
            "<|endoftext|>Hello World! My dear",
            "<|endoftext|>Some say we are tired far",
            "<|endoftext|>Hear that?",
        ]
        encoded = tokenizer.encode_batch(prompts, allowed_special="all")
        input_ids, segment_ids = pad_tokens(encoded, pad_to_power_of_two=True)
        key, subkey = jax.random.split(key)

        start = time.perf_counter()
        with jax.set_mesh(cfg.mesh):
            _, next_tokens, cache = prefill(
                model, input_ids, segment_ids, cache, head_dim, pad_id=PAD_ID
            )

        generated_tokens = (
            jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
            .at[:, 0]
            .set(next_tokens)
        )
        last_token = next_tokens[:, None]

        with jax.set_mesh(cfg.mesh):
            generated = generate(
                model,
                temp_cache,
                last_token,
                generated_tokens,
                head_dim,
                subkey,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )
        end = time.perf_counter()
        print(
            f"Time taken to generate {max_new_tokens * len(prompts)} tokens: {(end - start):.2f} seconds"
        )
        decoded = tokenizer.decode_batch(generated.tolist())

    elif MODEL_TYPE == "SFT":
        examples = {
            "messages": [
                {
                    "content": 'What are the benefits of regular exercise? Your response should contain at least 3 sentences. Include keywords such as "health", "reduce", and "improve".\n',
                    "role": "user",
                },
            ],
        }
        prompts = [
            format_conversation(ex, tok_info)["text"] + assistant_start
            for ex in examples
        ]
        encoded = tokenizer.encode_batch(prompts, allowed_special="all")
        input_ids, segment_ids = pad_tokens(encoded, pad_to_power_of_two=True)
        # Set the cache
        cache_key = jax.random.PRNGKey(1)
        batch_size = input_ids.shape[0]
        cache = KVCache.init(cache_key, cfg.mesh, cfg.rules, batch_size, cfg)

        key = jax.random.PRNGKey(123)
        key, subkey = jax.random.split(key)

        with jax.set_mesh(cfg.mesh):
            _, next_tokens, cache = prefill(
                model, input_ids, segment_ids, cache, head_dim, pad_id=PAD_ID
            )

        generated_tokens = (
            jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
            .at[:, 0]
            .set(next_tokens)
        )
        last_token = next_tokens[:, None]

        with jax.set_mesh(cfg.mesh):
            generated = generate_v2(
                model,
                cache,
                last_token,
                generated_tokens,
                head_dim,
                subkey,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                stop_token_id=tok_info["assistant_end_id"],
            )

        decoded = tokenizer.decode_batch(generated.tolist())

    for p, d in zip(prompts, decoded):
        print(f"Prompt: {p}")
        print(f"Completion: {d}")
        print("-" * 75)
