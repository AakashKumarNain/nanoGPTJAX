"""
This implementation of KVCache has been adopted and modified from https://github.com/jax-ml/jax-llm-examples
There are certain aspects of this implementation which the beginners may find unintuitive, especially the
rolling buffer implementation.

TODO: Make it more intuitive to understand.
"""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard, auto_axes

from utils import ParamSpec
from utils import ParamInitializer
from utils import jax_pytree_struct, layer_repr


@jax_pytree_struct
class KVCache(ParamInitializer):
    k: list[jax.Array]  # (batch_size, kv_heads, max_seq_len, head_dim)
    v: list[jax.Array]  # (batch_size, kv_heads, max_seq_len, head_dim)
    iter: jax.Array     # [] sequences are right-aligned for slice update performance    # fmt: off
    starts: jax.Array   # [batch_size]  sequences are right-aligned, we need start indices. # fmt: off
    time_axis: int = dataclasses.field(metadata=dict(static=True), default=2)
    size: int = dataclasses.field(metadata=dict(static=True), default=-1)

    @classmethod
    def param_specs(cls, batch_size, cfg):
        cache_spec = ParamSpec(
            shape=(
                batch_size,
                cfg.model.attn.kv_heads,
                cfg.model.seqlen,
                cfg.model.attn.head_dim,
            ),
            dtype=cfg.model.dtype,
            logical_axes=("batch", "attn_kv_heads", "sequence", "attn_head_dim"),
            initializer=jax.nn.initializers.zeros,
        )
        iter = ParamSpec(
            shape=(),
            dtype=jnp.int32,
            logical_axes=(),
            initializer=jax.nn.initializers.constant(-1),
        )
        starts = ParamSpec(
            shape=(batch_size,),
            dtype=jnp.int32,
            logical_axes=("batch",),
            initializer=jax.nn.initializers.zeros,
        )

        cache = KVCache(
            k=[cache_spec for _ in range(cfg.model.num_layers)],
            v=[cache_spec for _ in range(cfg.model.num_layers)],
            iter=iter,
            starts=starts,
            size=cfg.model.seqlen,
        )
        return cache

    def fill_len(self) -> jax.Array:
        return jnp.where(self.iter >= 0, (self.iter - self.starts) % self.size, 0)

    @property
    def buffers(self):
        return (self.k, self.v)

    @classmethod
    def init(cls, key, mesh, rules, batch_size, cfg):
        return cls._init_fn(key, mesh, rules, batch_size, cfg)

    def __repr__(self):
        return layer_repr(self)


def update_slice(x: jax.Array, y: jax.Array, pos: int, update_axis: int):
    y = reshard(y.astype(x.dtype), jax.typeof(x).sharding.spec)
    return jax.lax.dynamic_update_slice_in_dim(x, y, pos, axis=update_axis)


def make_attention_mask(
    q_len, k_len, q_segment_ids, kv_segment_ids, q_offset, kv_offset, causal: bool
):
    segment_mask = (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :])[
        :, None, :, :
    ]  # [B, 1, t, T]
    if causal:
        qk = (1, 1, q_len, k_len)  # [b, h, t, T]
        q_positions = (
            jax.lax.broadcasted_iota(jnp.int32, qk, 2) + q_offset[:, None, None, None]
        )
        kv_positions = (
            jax.lax.broadcasted_iota(jnp.int32, qk, 3) + kv_offset[:, None, None, None]
        ) % k_len
        causal_mask = q_positions >= kv_positions
        return segment_mask & causal_mask
    return segment_mask


@partial(jax.jit, static_argnums=(1, 2))
def prepare_chunk(tokens, pad_to: int, pad_id: int):
    """Left-pad token sequences to pad_to and emit binary mask (1=token)."""
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    padding_width = pad_to - tokens.shape[-1]
    tokens = jnp.pad(
        tokens, [(0, 0), (padding_width, 0)], mode="constant", constant_values=pad_id
    )
    segment_ids = jnp.where(tokens != pad_id, 1, 0).astype(jnp.int32)
    return tokens, segment_ids


@partial(auto_axes, out_sharding=P(None))
def count_left_padding(token_ids, pad_id):
    """Count leading pad tokens per batch row."""
    seen_token = jnp.cumsum(token_ids != pad_id, axis=-1)
    return jnp.sum(seen_token == 0, axis=-1)


@partial(auto_axes, out_sharding=P(None))
def length_minus_right_padding(segment_ids):
    """Count non-pad tokens ignoring trailing pad area."""
    reversed_tokens = jnp.flip(segment_ids != 0, axis=-1)
    seen = jnp.cumsum(reversed_tokens, axis=-1)
    return jnp.sum(seen > 0, axis=-1)


def segment_ids_to_positions(segment_ids):
    """Running position index for each contiguous non-zero segment."""

    def combine(prev, curr):
        prev_pos, prev_segment = prev
        curr_pos, curr_segment = curr
        same_segment = prev_segment == curr_segment
        next_pos = (prev_pos + 1) * same_segment + curr_pos
        return next_pos, curr_segment

    init_state = (jnp.zeros_like(segment_ids), segment_ids)
    combined = jax.lax.associative_scan(combine, init_state, axis=-1)
    return combined[0].astype(jnp.int32)
