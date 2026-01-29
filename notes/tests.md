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

- We still have not applied any tricks, yet the throughput, the optimization, and the model outputs all are in great shape.
- I am not sure how important the document boundary is for pretraining. As of now, the grain processes takes 500MB on
each GPU which is not good, and because of the process finding bos tokens and generating sequences on the fly, there
are times when the GPU utilization becomes poor. Though it's not that bad, but I would love to get rid of any bubble in the data loading pipeline. 


## Optimizer: AdamW and Muon

- Both adamw and muon works well. We achieve the same validation loss as achieved by `nanochat` and `modded-nanogpt`.
- Though it is straightforward to use muon in JAX (thanks to Optax!), but there are a few nuances that one needs to be aware of. Muon can be applied to any high rank array, but we need to make it aware of those dimensions, otherwise those leaves won't get muon benefits. For example, in our codebase the attention weights are 3D arrays as opposed to 2D. The reason we have kept them in 3D is because sharding becomes extremely easy. A side effect of this is that we need to
make muon aware of this to ensure that arrays are rehsape properly for orthogonalization, otheriwse they are assumed to be 2D.

Here is an example:

```python
ef make_muon(lr_schedule, weight_decay=0.0):
    def muon_dims_fn(p):
        def choose(x):
            s = getattr(x, "shape", None)
            if s is None:
                return None
            if len(s) == 2:
                return optax.contrib.MuonDimensionNumbers((0,), (1,))
            if len(s) == 3:
                if s[-1] == d_model:
                    return optax.contrib.MuonDimensionNumbers((0, 1), (2,))
                return optax.contrib.MuonDimensionNumbers((0,), (1, 2))
            return None
        return jax.tree_util.tree_map(choose, p)

    def wd_mask_fn(p):
        def keep(x):
            s = getattr(x, "shape", None)
            return s is not None and len(s) >= 2
        return jax.tree_util.tree_map(keep, p)

    optim = optax.contrib.muon(
        learning_rate=lr_schedule,
        ns_coeffs=(3.4445, -4.775, 2.0315),
        ns_steps=5,
        beta=b2,
        eps=1e-8,
        weight_decay=weight_decay,
        weight_decay_mask=wd_mask_fn,
        mu_dtype=jnp.float32,
        nesterov=True,
        adaptive=False,
        adam_b1=b1,
        adam_b2=b2,
        adam_eps_root=0.0,
        adam_weight_decay=weight_decay,
        adam_learning_rate=None,
        muon_weight_dimension_numbers=muon_dims_fn,
    )
```

Though the above implementation works perfectly and converges very quickly compared to `AdamW`, we take a big hit on the throughput. For example, if a single
H100 instance was processing almost 400K tokens/second with `adamw`, switching to the above will reduce the throughput to 300-330K tokens/second.
That's a big hit!

Why the drop, one may ask? The `Newton–Schulz orthogonalization` adds extra matmuls, and if you flatten across a sharded axis it also triggers extra cross‑device collectives. Both effects reduce tokens/s by roughly 5–15% in practice. An easy to get back close to the original throughput is to ensure Muon orthogonalize “locally” by treating the sharded head axis as a batch axis (so no collectives across heads). For example, `wq/wk/wv (d_emb, heads, head_dim): use reduction=(0,), output=(2,)` and leave heads as batch. Similarly, `wo (heads, head_dim, d_emb): use reduction=(1,), output=(2,)` and leave heads as batch. This still gives the intended 2D matrix per head, but avoids flattening across heads. Optax supports this directly via muon_weight_dimension_numbers. We can
also bring down Newton–Schulz iterations to 3. 

```python
def make_weight_dim_nums(p):
    def choose(x):
        s = getattr(x, "shape", None)
        if s is None:
            return None
        if len(s) == 2:
            return optax.contrib.MuonDimensionNumbers((0,), (1,))
        if len(s) == 3:
            if s[-1] == d_model: # wo: (heads, head_dim, d_model)
                return optax.contrib.MuonDimensionNumbers((1,), (2,))
            return optax.contrib.MuonDimensionNumbers((0,), (2,))  # wq/wk/wv: batch=heads
        return None
    return jax.tree_util.tree_map(choose, p)


def weight_decay_mask_fn(p):
    def keep(x):
        s = getattr(x, "shape", None)
        return s is not None and len(s) >= 2
    return jax.tree_util.tree_map(keep, p)


muon_weight_dim_nums = make_weight_dim_nums(params)
muon_wd_mask = weight_decay_mask_fn(params)

def make_muon(lr_schedule, weight_decay=0.0):
    return optax.contrib.muon(
        learning_rate=lr_schedule,
        ns_coeffs=(3.4445, -4.775, 2.0315),
        ns_steps=3,
        beta=b2,
        eps=1e-8,
        weight_decay=weight_decay,
        weight_decay_mask=muon_wd_mask,
        mu_dtype=jnp.float32,
        nesterov=True,
        adaptive=False,
        adam_b1=b1,
        adam_b2=b2,
        adam_eps_root=0.0,
        adam_weight_decay=weight_decay,
        adam_learning_rate=None,
        muon_weight_dimension_numbers=muon_dims_fn,
    )
```