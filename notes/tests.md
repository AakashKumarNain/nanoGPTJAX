The new kvcache implementation for forward pass where prompts are left-padded, and generated sequences are right-aligned works well without jit. JIT also
works but with `cudnn` attention implementation right now, it's throwing some error that corresponds to some layout issues. It works well with other `xla` or `None`, but after a point the output diverges from the actual output obtained with `cudnn`. This issue has already been raised with the JAX team.

For now, here are the new version of prefill, and decode we have to include in the code once we figure out to deal with the errors. These work perfectly without JIT. Take utmost care when expanding the dimensions after prefill. We always have to expand the last dimension.

```python
devices = np.array(jax.devices())
mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
cfg = Config(mesh=mesh, rules=sharding_rules)
tokenizer = tiktoken.get_encoding("gpt2")

ckpt_path = "/jaxnano/ckpts/exp/4700/params"
model_sharding = GPT.shardings(cfg.mesh, cfg.rules, cfg.model)
model = load_weights_from_checkpoint(ckpt_path, model_sharding)
print("Weights loaded from the checkpoint successfully!")


tokenizer = tiktoken.get_encoding("gpt2")
PAD_TOKEN = "<|pad|>"
tokenizer = tiktoken.Encoding(
    name="gpt2_with_pad",
    pat_str=tokenizer._pat_str,
    mergeable_ranks=tokenizer._mergeable_ranks,
    special_tokens={
        **tokenizer._special_tokens,
        PAD_TOKEN: tokenizer.n_vocab,  # next available token id
    },
)

PAD_ID = tokenizer.encode(PAD_TOKEN,  allowed_special={"<|endoftext|>", "<|pad|>"})[0]

seqlen = cfg.model.seqlen
head_dim = cfg.model.attn.head_dim
jax.set_mesh(cfg.mesh)
freqs = precompute_frequencies(positions=jnp.arange(seqlen)[None, :], features=head_dim)

rompts = [
        "<|endoftext|>Did you notice that this world",
        "<|endoftext|>Hear that?",
    ]


def pad_tokens(tokens, pad_id=PAD_ID, pad_to_power_of_two=False):
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


# @partial(jax.jit, static_argnames=("head_dim", "pad_id"))
def prefill(params, input_ids, segment_ids, cache, head_dim, pad_id=PAD_ID):
    left_pad_counts = count_left_padding(input_ids, pad_id=pad_id)
    uninitialized_iter = -jnp.ones_like(cache.iter)
    cache = dataclasses.replace(
        cache, 
        starts=left_pad_counts, 
        iter=uninitialized_iter
    )
    
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    
    # With left padding, last valid token is always at position -1 (rightmost)
    last_token_logits = logits[:, -1, :]
    next_tokens = jnp.argmax(last_token_logits, axis=-1)
    
    return logits, next_tokens, cache

# @partial(jax.jit, static_argnames=("head_dim",))
def decode(params, input_ids, cache, head_dim):
    segment_ids = jnp.ones_like(input_ids, dtype=jnp.int32)
    logits, cache = forward_v2(params, input_ids, segment_ids, cache, head_dim)
    # TODO: Add sampling here
    # next_tokens = jnp.argmax(logits[:, -1, :], axis=-1)
    return logits[:, -1, :], cache

encoded = tokenizer.encode_batch(prompts, allowed_special={"<|endoftext|>", "<|pad|>"})
input_ids, segment_ids = pad_tokens(encoded, pad_to_power_of_two=True)

# Test prefill
cache_key = jax.random.PRNGKey(1)
batch_size = input_ids.shape[0]
cache = KVCache.init(cache_key, cfg.mesh, cfg.rules, batch_size, cfg)
logits, next_tokens, cache = prefill(model, input_ids, segment_ids, cache, head_dim, pad_id=PAD_ID)
print(next_tokens, tokenizer.decode(next_tokens))
# Should output: [318 921]  is You

# Test decode
for _ in range(10):
    if next_tokens.ndim == 1:
        next_tokens = next_tokens[:, None]
    logits, cache = decode(model, next_tokens, cache, head_dim)
    next_tokens = jnp.argmax(logits, -1)
    print(next_tokens)

# These should be the 11 tokens we have generated so far for these two prompts
# print(tokenizer.decode([3128, 257, 1310, 1180, 422, 262, 1334, 286, 262, 995, 30]))
# print(tokenizer.decode([921, 447, 247, 260, 407, 3436, 13, 198, 464, 717, 640]))

```

We got rid of the above bug by explicitly copying the `k` and `v` values into new arrays. Without creating an extra copy, it is not possible
to make the array contiguous as transpose is just a view in the end. It works now. 


---

# Optimizations

- Both adamw and muon works well. We achieve the same validation loss as achieved by `nanochat` and `modded-nanogpt`.
- We still have not applied any tricks, yet the throughput, the optimization, and the model outputs all are in great shape.