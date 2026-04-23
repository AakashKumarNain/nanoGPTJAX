import math
import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from kvcache import update_slice
from kvcache import compute_segment_mask
from kvcache import make_attention_mask
from kvcache import segment_ids_to_positions

from utils import layer_repr
from utils import ParamInitializer
from utils import jax_pytree_struct
from layers import Embedding, Linear, GroupedQueryAttention
from quantization import QArray, maybe_dequantize


if jax.default_backend() == "gpu":
    ATTN_IMPL = "cudnn"
elif jax.default_backend() == "tpu":
    ATTN_IMPL = "xla"
else:
    ATTN_IMPL = None


def compute_layer_configs(window_pattern, num_layers, short_window, nope_global=True):
    """Compute per-layer (local_window_size, use_rope) from a window pattern string.

    Args:
        window_pattern: String like "SSSL". S = short/local, L = long/global.
            Tiled across layers.
            "L" alone = all layers use full attention with RoPE (original model).
            "S" alone = all layers use SWA with RoPE, no global attention.
            Mixed patterns (e.g. "SL", "SSSL") = tiled as given, with last
                layer forced to global NoPE attention.
        num_layers: Total number of transformer layers.
        short_window: Number of past tokens local layers attend to.
            e.g. 4096 means attend to 4096 tokens before self + self.
        nope_global: If True, global (L) layers skip RoPE (paper's approach).
            If False, all layers use RoPE (nanochat's approach).

    Returns:
        local_window_sizes: List of (left, right) tuples or None per layer.
        use_rope_flags: List of bools per layer.
    """

    pattern = window_pattern.upper()
    for char in pattern:
        if char not in ("S", "L"):
            raise ValueError(
                f"Invalid character '{char}' in window_pattern. Use 'S' or 'L'."
            )

    # "L" only: all full attention with RoPE
    if set(pattern) == {"L"}:
        return [None] * num_layers, [True] * num_layers

    # "S" only: all SWA with RoPE, no global attention
    if pattern == "S":
        return [(short_window, 0)] * num_layers, [True] * num_layers

    # Mixed pattern: tile across layers, last layer forced to global NoPE
    local_window_sizes = []
    rope_flags = []
    for i in range(num_layers):
        char = pattern[i % len(pattern)]
        if char == "S":
            # Use SWA with RoPE
            local_window_sizes.append((short_window, 0))
            rope_flags.append(True)
        else:
            # Use global attention with Nope
            local_window_sizes.append(None)
            rope_flags.append(not nope_global)

    # Irrespective of what the user passes, for stability, we always use
    # global attention for the last layer if a mixed pattern is passed.
    local_window_sizes[-1] = None
    rope_flags[-1] = not nope_global
    return local_window_sizes, rope_flags


@jax_pytree_struct
class MLP(ParamInitializer):
    fc1: Linear
    fc2: Linear

    @classmethod
    def param_specs(cls, cfg):
        fc1 = Linear.param_specs(cfg.fc1)
        fc2 = Linear.param_specs(cfg.fc2)
        return MLP(fc1=fc1, fc2=fc2)

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class TransformerBlock(ParamInitializer):
    attn: GroupedQueryAttention
    mlp: MLP

    @classmethod
    def param_specs(cls, cfg):
        attn = GroupedQueryAttention.param_specs(cfg.attn)
        mlp = MLP.param_specs(cfg.mlp)
        return TransformerBlock(attn=attn, mlp=mlp)

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class GPT(ParamInitializer):
    embed: Embedding
    blocks: list[TransformerBlock]
    lm_head: Linear
    local_window_sizes: tuple = dataclasses.field(
        metadata=dict(static=True), default=None
    )
    rope_flags: tuple = dataclasses.field(metadata=dict(static=True), default=None)

    @classmethod
    def param_specs(cls, cfg):
        embed = Embedding.param_specs(cfg.embed)
        blocks = [TransformerBlock.param_specs(cfg) for _ in range(cfg.attn.num_layers)]
        lm_head = Linear.param_specs(cfg.lm_head)
        local_window_sizes, rope_flags = compute_layer_configs(
            cfg.window_pattern,
            cfg.attn.num_layers,
            cfg.local_window_size,
            cfg.nope_global_attn,
        )
        return GPT(
            embed=embed,
            blocks=blocks,
            lm_head=lm_head,
            local_window_sizes=tuple(local_window_sizes),
            rope_flags=tuple(rope_flags),
        )

    @classmethod
    def init(cls, key, cfg, **kwargs):
        return cls._init_fn(key, cfg.mesh, cfg.rules, cfg.model, **kwargs)

    def __repr__(self):
        return layer_repr(self)


def count_params(model):
    """Count the parameters in an Equinox model"""
    leaves = jax.tree_util.tree_leaves(
        model, is_leaf=lambda x: x is None or isinstance(x, QArray)
    )
    total = 0
    for leaf in leaves:
        if leaf is None:
            continue
        if isinstance(leaf, QArray):
            if hasattr(leaf.qvalue, "size"):
                total += leaf.qvalue.size
            else:
                total += math.prod(leaf.qvalue.shape)
        else:
            total += leaf.size
    return total


def precompute_frequencies(
    positions: jax.Array, features: int, theta=10000.0, dtype=None
):
    """Generate Sin/Cos for Rotary Embeddings."""
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = theta**fraction
    rotational_frequency = 1.0 / timescale
    sinusoid_inp = jnp.einsum(
        "BT,k->BTk",
        positions,
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
        out_sharding=P(None, None, None),
    )
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    if dtype is not None:
        sin = sin.astype(dtype)
        cos = cos.astype(dtype)
    return sin, cos


def calculate_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    assert x.ndim == 4 and sin.ndim == 3 and cos.ndim == 3
    orig_dtype = x.dtype
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    sin, cos = sin[:, :, None, :], cos[:, :, None, :]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(
        orig_dtype
    )


def embedding_forward(params, x):
    weight = maybe_dequantize(params.weight)
    return weight.at[x, :].get()


def rmsnorm_forward(x, eps=1e-5):
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x / scale).astype(orig_dtype)


def linear_forward(params, x):
    weight = maybe_dequantize(params.weight)
    out = jnp.einsum("...d, dv-> ...v", x, weight)
    if params.bias is not None:
        return out + params.bias
    else:
        return out


def mlp_forward(params, x):
    x = linear_forward(params.fc1, x)
    x = jnp.square(jax.nn.relu(x))
    x = linear_forward(params.fc2, x)
    return x


# Though we can combine the cache update this in this block. I keep two versions
# of attention_forward and block_forward because it lets me optimize whatever
# cache implementation I want during the inference time without changing the
# training code. This lets me keep running my experiments in parallel. Redundancy
# in this case is good.

#################################### For training ########################################


def attn_forward(params, x, mask, freqs, use_rope, local_window_size):
    orig_dtype = x.dtype
    sin, cos = freqs

    with jax.named_scope("qkv_matmul"):
        wq = maybe_dequantize(params.wq)
        wk = maybe_dequantize(params.wk)
        wv = maybe_dequantize(params.wv)
        q = jnp.einsum("btd, dhq -> bthq", x, wq)
        k = jnp.einsum("btd, dhq -> bthq", x, wk)
        v = jnp.einsum("btd, dhq -> bthq", x, wv)

    with jax.named_scope("qk_norm"):
        q = rmsnorm_forward(q)
        k = rmsnorm_forward(k)

    if use_rope:
        with jax.named_scope("rope"):
            q = calculate_rope(q, sin, cos)
            k = calculate_rope(k, sin, cos)

    with jax.named_scope("attention"):
        scale = 1.0 / math.sqrt(q.shape[-1])
        if mask is not None:
            attn = jax.nn.dot_product_attention(
                q,
                k,
                v,
                mask=mask,
                scale=scale,
                is_causal=True,
                local_window_size=local_window_size,
                implementation=ATTN_IMPL,
            ).astype(orig_dtype)
        else:
            attn = jax.nn.dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                is_causal=True,
                local_window_size=local_window_size,
                implementation=ATTN_IMPL,
            ).astype(orig_dtype)

    with jax.named_scope("projection"):
        wo = maybe_dequantize(params.wo)
        out = jnp.einsum("bthq, hqd->btd", attn, wo)
    return out


def block_forward(params, x, mask, freqs, use_rope, local_window_size):
    with jax.named_scope("pre_attn_norm"):
        attn_in = rmsnorm_forward(x)

    attn_out = attn_forward(
        params.attn, attn_in, mask, freqs, use_rope, local_window_size
    )

    with jax.named_scope("residual"):
        x = x + attn_out

    with jax.named_scope("post_attn_norm"):
        ffn_in = rmsnorm_forward(x)

    with jax.named_scope("ffn"):
        ffn_out = mlp_forward(params.mlp, ffn_in)

    with jax.named_scope("residual"):
        x = x + ffn_out
    return x


def forward(params, x, segment_ids, freqs):
    if segment_ids is not None:
        with jax.named_scope("compute_mask"):
            mask = compute_segment_mask(segment_ids)
    else:
        mask = None

    with jax.named_scope("embedding"):
        x = embedding_forward(params.embed, x)

    for idx, block in enumerate(params.blocks):
        x = block_forward(
            block,
            x,
            mask,
            freqs,
            params.rope_flags[idx],
            params.local_window_sizes[idx],
        )

    with jax.named_scope("norm"):
        x = rmsnorm_forward(x)

    with jax.named_scope("unembed"):
        logits = linear_forward(params.lm_head, x)

    with jax.named_scope("logit_soft_capping"):
        logits = logits.astype(jnp.float32)
        logits = 15.0 * jnp.tanh(logits / 15.0)
    return logits


############## Inference: with new simplified KVCache implementation ##############


def attn_forward_infer(
    params,
    x,
    q_segment_ids,
    q_positions,
    freqs,
    cache,
    idx,
    use_rope,
    local_window_size,
):
    orig_dtype = x.dtype
    sin, cos = freqs

    with jax.named_scope("qkv_matmul"):
        wq = maybe_dequantize(params.wq)
        wk = maybe_dequantize(params.wk)
        wv = maybe_dequantize(params.wv)
        q = jnp.einsum("btd, dhq -> bthq", x, wq)
        k = jnp.einsum("btd, dhq -> bthq", x, wk)
        v = jnp.einsum("btd, dhq -> bthq", x, wv)

    with jax.named_scope("qk_norm"):
        q = rmsnorm_forward(q)
        k = rmsnorm_forward(k)

    if use_rope:
        with jax.named_scope("rope"):
            q = calculate_rope(q, sin, cos)
            k = calculate_rope(k, sin, cos)

    with jax.named_scope("cache_update"):
        write_pos = cache.next_write_index()
        chunk_len = jnp.asarray(x.shape[1], dtype=cache.end.dtype)

        k_updated = update_slice(
            cache.k[idx], k, write_pos, update_axis=cache.time_axis
        )
        v_updated = update_slice(
            cache.v[idx], v, write_pos, update_axis=cache.time_axis
        )
        cache_updates = (k_updated, v_updated)

        end_after_write = cache.end + chunk_len
        kv_positions = (
            cache.slot_positions(end=end_after_write) - cache.left_pad[:, None]
        )
        kv_segment_ids = cache.slot_segment_ids(end=end_after_write)

    with jax.named_scope("attention"):
        scale = 1.0 / math.sqrt(q.shape[-1])
        mask = make_attention_mask(
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
            causal=True,
            local_window_size=local_window_size,
        )

        attn = jax.nn.dot_product_attention(
            q,
            k_updated,
            v_updated,
            mask=mask,
            implementation=ATTN_IMPL,
            scale=scale,
        ).astype(orig_dtype)

    with jax.named_scope("projection"):
        wo = maybe_dequantize(params.wo)
        out = jnp.einsum("bthq, hqd->btd", attn, wo)
    return out, cache_updates


def block_forward_infer(
    params,
    x,
    q_segment_ids,
    q_positions,
    freqs,
    cache,
    idx,
    use_rope,
    local_window_size,
):
    with jax.named_scope("pre_attn_norm"):
        attn_in = rmsnorm_forward(x)

    attn_out, cache_updates = attn_forward_infer(
        params.attn,
        attn_in,
        q_segment_ids,
        q_positions,
        freqs,
        cache,
        idx,
        use_rope,
        local_window_size,
    )

    with jax.named_scope("residual"):
        x = x + attn_out

    with jax.named_scope("post_attn_norm"):
        ffn_in = rmsnorm_forward(x)

    with jax.named_scope("ffn"):
        ffn_out = mlp_forward(params.mlp, ffn_in)

    with jax.named_scope("residual"):
        x = x + ffn_out
    return x, cache_updates


def forward_infer(params, x, segment_ids, cache, head_dim):
    chunk_len = jnp.asarray(x.shape[1], dtype=cache.end.dtype)
    q_segment_ids = jnp.where(segment_ids != 0, 1, 0).astype(jnp.int32)
    positions = segment_ids_to_positions(segment_ids) + cache.fill_len()[:, None]

    with jax.named_scope("embedding"):
        x = embedding_forward(params.embed, x)

    freqs = precompute_frequencies(positions, features=head_dim, dtype=x.dtype)

    all_cache_updates = []
    for idx, block in enumerate(params.blocks):
        x, cache_updates = block_forward_infer(
            block,
            x,
            q_segment_ids,
            positions,
            freqs,
            cache,
            idx,
            params.rope_flags[idx],
            params.local_window_sizes[idx],
        )
        all_cache_updates.append(cache_updates)

    with jax.named_scope("norm"):
        x = rmsnorm_forward(x)

    with jax.named_scope("unembed"):
        logits = linear_forward(params.lm_head, x)

    with jax.named_scope("logit_soft_capping"):
        logits = 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)

    new_k = [z[0] for z in all_cache_updates]
    new_v = [z[1] for z in all_cache_updates]
    cache = dataclasses.replace(cache, k=new_k, v=new_v, end=cache.end + chunk_len)
    return logits, cache
