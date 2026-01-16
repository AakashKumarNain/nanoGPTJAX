import math
import dataclasses

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from kvcache import update_slice
from kvcache import count_left_padding
from kvcache import make_attention_mask
from kvcache import length_minus_right_padding
from kvcache import segment_ids_to_positions

from utils import layer_repr
from utils import ParamInitializer
from utils import jax_pytree_struct
from layers import Embedding, Linear, GroupedQueryAttention


if jax.default_backend() == "gpu":
    ATTN_IMPL = "cudnn"
elif jax.default_backend() == "tpu":
    ATTN_IMPL = "xla"
else:
    ATTN_IMPL = None


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

    @classmethod
    def param_specs(cls, cfg):
        embed = Embedding.param_specs(cfg.embed)
        blocks = [TransformerBlock.param_specs(cfg) for _ in range(cfg.attn.num_layers)]
        lm_head = Linear.param_specs(cfg.lm_head)
        return GPT(embed=embed, blocks=blocks, lm_head=lm_head)

    @classmethod
    def init(cls, key, cfg):
        return cls._init_fn(key, cfg.mesh, cfg.rules, cfg.model)

    def __repr__(self):
        return layer_repr(self)


def count_params(model):
    """Count the parameters in an Equinox model"""
    return sum(x.size for x in jax.tree_util.tree_leaves(model))


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
    return params.weight.at[x, :].get()


def rmsnorm_forward(x, eps=1e-5):
    orig_dtype = x.dtype
    x = x.astype(jnp.float32)
    scale = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps)
    return (x / scale).astype(orig_dtype)


def linear_forward(params, x):
    out = jnp.einsum("...d, dv-> ...v", x, params.weight)
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


def attn_forward(params, x, freqs):
    orig_dtype = x.dtype
    bsz, seqlen, _ = x.shape
    sin, cos = freqs

    with jax.named_scope("qkv_matmul"):
        q = jnp.einsum("btd, dhq -> bthq", x, params.wq)
        k = jnp.einsum("btd, dhq -> bthq", x, params.wk)
        v = jnp.einsum("btd, dhq -> bthq", x, params.wv)

    with jax.named_scope("qk_norm"):
        q = rmsnorm_forward(q)
        k = rmsnorm_forward(k)

    with jax.named_scope("rope"):
        q = calculate_rope(q, sin, cos)
        k = calculate_rope(k, sin, cos)

    # output shape: (bsz, seqlen, q_heads, head_dim)
    with jax.named_scope("attention"):
        scale = None  # 1.0 / math.sqrt(q.shape[-1])
        attn = jax.nn.dot_product_attention(
            q, k, v, is_causal=True, implementation=ATTN_IMPL, scale=scale
        ).astype(orig_dtype)

    # params.wo is of shape (q_heads, head_dim, d_emb)
    with jax.named_scope("projection"):
        out = jnp.einsum("bthq, hqd->btd", attn, params.wo)
    return out


def block_forward(params, x, freqs):
    # Attention
    with jax.named_scope("pre_attn_norm"):
        attn_in = rmsnorm_forward(x)

    attn_out = attn_forward(params.attn, attn_in, freqs)

    with jax.named_scope("residual"):
        x = x + attn_out

    # FFN
    with jax.named_scope("post_attn_norm"):
        ffn_in = rmsnorm_forward(x)
    with jax.named_scope("ffn"):
        ffn_out = mlp_forward(params.mlp, ffn_in)
    with jax.named_scope("residual"):
        x = x + ffn_out
    return x


def forward(params, x, freqs):
    with jax.named_scope("embedding"):
        x = embedding_forward(params.embed, x)

    for block in params.blocks:
        x = block_forward(block, x, freqs)

    with jax.named_scope("norm"):
        x = rmsnorm_forward(x)

    with jax.named_scope("unembed"):
        logits = linear_forward(params.lm_head, x)

    with jax.named_scope("logit_soft_capping"):
        logits = logits.astype(jnp.float32)
        logits = 15.0 * jnp.tanh(logits / 15.0)
    return logits


#################################### For inference ########################################


def attn_forward_v2(params, x, segment_ids, freqs, cache, idx):
    orig_dtype = x.dtype
    sin, cos = freqs

    # 1. QKV Projection: (B, T, D) -> (B, T, H, D)
    with jax.named_scope("qkv_matmul"):
        q = jnp.einsum("btd, dhq -> bthq", x, params.wq)
        k = jnp.einsum("btd, dhq -> bthq", x, params.wk)
        v = jnp.einsum("btd, dhq -> bthq", x, params.wv)

    # 2. Norm: (B, T, H, D)
    with jax.named_scope("qk_norm"):
        q = rmsnorm_forward(q)
        k = rmsnorm_forward(k)

    # 3. RoPE: (B, T, H, D)
    with jax.named_scope("rope"):
        q = calculate_rope(q, sin, cos)
        k = calculate_rope(k, sin, cos)

    # 4. Cache updates. Cache shape: (B, H, T, D)
    with jax.named_scope("cache_update"):
        kt = jnp.transpose(k, (0, 2, 1, 3))
        vt = jnp.transpose(v, (0, 2, 1, 3))
        it = jnp.maximum(cache.iter, 0)

        k_updated = update_slice(cache.k[idx], kt, it, update_axis=cache.time_axis)
        v_updated = update_slice(cache.v[idx], vt, it, update_axis=cache.time_axis)
        cache_updates = (k_updated, v_updated)

        # fmt: off
        # Compute masks using original logic
        additional_tokens = jnp.max(length_minus_right_padding(segment_ids))
        time_indices = (jnp.arange(0, v_updated.shape[-2])[None, :] - cache.starts[:, None]) % cache.size
        q_segment_ids = jnp.where(segment_ids != 0, 1, 0)
        kv_segment_ids = (time_indices >= 0) & (time_indices < cache.fill_len()[:, None] + additional_tokens)
        q_offset = cache.fill_len() - count_left_padding(q_segment_ids, pad_id=0)
        kv_offset = -cache.starts

        # If we directly use transpose in attention, it will throw an error
        # because the array layout is not contiguous. In JAX, the only way
        # to ensure that arrays are contagious is to make an **explicit** copy.
        kt = jnp.copy(jnp.transpose(k_updated, (0, 2, 1, 3)))
        vt = jnp.copy(jnp.transpose(v_updated, (0, 2, 1, 3)))
        # fmt: on

    # 5. Attention
    with jax.named_scope("attention"):
        qlen = q.shape[1]
        kvlen = k_updated.shape[2]
        scale = 1.0 / math.sqrt(q.shape[-1])
        mask = make_attention_mask(
            qlen, kvlen, q_segment_ids, kv_segment_ids, q_offset, kv_offset, causal=True
        )

        # fmt: off
        attn = jax.nn.dot_product_attention(
            q,          # (B, T, H, D)
            kt,         # (B, S, H, D)
            vt,         # (B, S, H, D)
            mask=mask,  # (B, 1, T, S)
            implementation=ATTN_IMPL,
            scale=scale
        ).astype(orig_dtype)
        # fmt: on

    # 6. Projection
    with jax.named_scope("projection"):
        out = jnp.einsum("bthq, hqd->btd", attn, params.wo)
    return out, cache_updates


def block_forward_v2(params, x, segment_ids, freqs, cache, idx):
    with jax.named_scope("pre_attn_norm"):
        attn_in = rmsnorm_forward(x)

    attn_out, cache_updates = attn_forward_v2(
        params.attn, attn_in, segment_ids, freqs, cache, idx
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


def forward_v2(params, x, segment_ids, cache, head_dim):
    with jax.named_scope("embedding"):
        x = embedding_forward(params.embed, x)

    positions = segment_ids_to_positions(segment_ids)
    positions = positions + cache.fill_len()[:, None]
    freqs = precompute_frequencies(positions, features=head_dim, dtype=x.dtype)

    all_cache_updates = []
    for idx, block in enumerate(params.blocks):
        x, cache_updates = block_forward_v2(block, x, segment_ids, freqs, cache, idx)
        all_cache_updates.append(cache_updates)

    with jax.named_scope("norm"):
        x = rmsnorm_forward(x)

    with jax.named_scope("unembed"):
        logits = linear_forward(params.lm_head, x)

    with jax.named_scope("logit_soft_capping"):
        logits = 15.0 * jnp.tanh(logits.astype(jnp.float32) / 15.0)

    # Update cache
    new_k = [z[0] for z in all_cache_updates]
    new_v = [z[1] for z in all_cache_updates]
    additional_tokens = jnp.max(length_minus_right_padding(segment_ids))
    cache = dataclasses.replace(
        cache,
        k=new_k,
        v=new_v,
        iter=(jnp.maximum(0, cache.iter) + additional_tokens) % cache.size,
    )
    return logits, cache
