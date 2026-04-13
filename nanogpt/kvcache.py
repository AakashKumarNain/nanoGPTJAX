"""
Cleaner rolling KV-cache for full-attention decode.

This version keeps an absolute write cursor instead of mixing a wrapped cursor
with per-row starts. The physical buffer is still circular, but the logical
position math is explicit:

- `end` is the next absolute position to write.
- `left_pad` stores the number of left-pad tokens per batch row for prefill.
- physical slot -> logical position mapping is derived on demand.

The goal here is not a drop-in replacement for every current call site. The
goal is a cache that is easier to reason about and safer to extend.
"""

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import auto_axes, reshard

from utils import ParamInitializer
from utils import ParamSpec
from utils import jax_pytree_struct, layer_repr


@jax_pytree_struct
class KVCache(ParamInitializer):
    k: list[jax.Array]  # (batch_size, max_seq_len, kv_heads, head_dim)
    v: list[jax.Array]  # (batch_size, max_seq_len, kv_heads, head_dim)
    end: jax.Array  # [] absolute next write position
    left_pad: jax.Array  # [batch_size] count of left-pad tokens used at prefill
    time_axis: int = dataclasses.field(metadata=dict(static=True), default=1)
    size: int = dataclasses.field(metadata=dict(static=True), default=-1)

    @classmethod
    def param_specs(cls, batch_size, cfg):
        cache_spec = ParamSpec(
            shape=(
                batch_size,
                cfg.model.seqlen,
                cfg.model.attn.kv_heads,
                cfg.model.attn.head_dim,
            ),
            dtype=cfg.model.dtype,
            logical_axes=("batch", "sequence", "kv_heads", "head_dim"),
            initializer=jax.nn.initializers.zeros,
        )
        end = ParamSpec(
            shape=(),
            dtype=jnp.int32,
            logical_axes=(),
            initializer=jax.nn.initializers.zeros,
        )
        left_pad = ParamSpec(
            shape=(batch_size,),
            dtype=jnp.int32,
            logical_axes=("batch",),
            initializer=jax.nn.initializers.zeros,
        )

        return KVCache(
            k=[cache_spec for _ in range(cfg.model.num_layers)],
            v=[cache_spec for _ in range(cfg.model.num_layers)],
            end=end,
            left_pad=left_pad,
            size=cfg.model.seqlen,
        )

    @classmethod
    def init(cls, key, mesh, rules, batch_size, cfg):
        return cls._init_fn(key, mesh, rules, batch_size, cfg)

    @property
    def buffers(self):
        return (self.k, self.v)

    @property
    def batch_size(self) -> int:
        return self.left_pad.shape[0]

    def next_write_index(self, end=None) -> jax.Array:
        end = self.end if end is None else jnp.asarray(end, dtype=self.end.dtype)
        return jnp.mod(end, self.size)

    def oldest_cached_position(self, end=None) -> jax.Array:
        end = self.end if end is None else jnp.asarray(end, dtype=self.end.dtype)
        return jnp.maximum(end - self.size, 0)

    def fill_len(self, end=None) -> jax.Array:
        end = self.end if end is None else jnp.asarray(end, dtype=self.end.dtype)
        return jnp.clip(end - self.left_pad, 0, self.size)

    def is_full(self, end=None) -> jax.Array:
        end = self.end if end is None else jnp.asarray(end, dtype=self.end.dtype)
        return end >= self.size

    def with_left_padding(self, left_pad: jax.Array):
        left_pad = jnp.asarray(left_pad, dtype=self.left_pad.dtype)
        return dataclasses.replace(
            self, left_pad=left_pad, end=jnp.zeros_like(self.end)
        )

    def advance(self, token_count):
        token_count = jnp.asarray(token_count, dtype=self.end.dtype)
        return dataclasses.replace(self, end=self.end + token_count)

    def slot_positions(self, end=None) -> jax.Array:
        """Absolute position represented by each physical slot.

        Returns:
            Array of shape [batch_size, size]. Absolute positions are identical
            across batch rows; validity still depends on `left_pad`.
        """
        end = self.end if end is None else jnp.asarray(end, dtype=self.end.dtype)
        slots = jnp.arange(self.size, dtype=self.end.dtype)
        oldest = self.oldest_cached_position(end=end)
        next_write = self.next_write_index(end=end)

        wrapped_positions = oldest + jnp.mod(slots - next_write, self.size)
        unwrapped_positions = slots
        positions = jnp.where(end > self.size, wrapped_positions, unwrapped_positions)
        return jnp.broadcast_to(positions[None, :], (self.batch_size, self.size))

    def slot_segment_ids(self, end=None) -> jax.Array:
        """Binary validity mask for cached KV slots, shape [batch_size, size]."""
        end = self.end if end is None else jnp.asarray(end, dtype=self.end.dtype)
        slot_positions = self.slot_positions(end=end)
        valid = (slot_positions >= self.left_pad[:, None]) & (slot_positions < end)
        return valid.astype(jnp.int32)

    def query_positions(self, q_len: int, start=None) -> jax.Array:
        """Absolute positions for the current query chunk, shape [batch_size, q_len]."""
        start = self.end if start is None else jnp.asarray(start, dtype=self.end.dtype)
        positions = start + jnp.arange(q_len, dtype=self.end.dtype)
        return jnp.broadcast_to(positions[None, :], (self.batch_size, q_len))

    def attention_metadata(self, segment_ids: jax.Array):
        """Metadata needed to build an attention mask after staging a write.

        The returned key/value positions and valid KV mask assume the current
        query chunk has already been written into the rolling buffer, while the
        query positions still correspond to the chunk's original absolute
        positions.
        """
        appended_tokens = jnp.max(length_minus_right_padding(segment_ids))
        q_segment_ids = jnp.where(segment_ids != 0, 1, 0).astype(jnp.int32)
        q_positions = self.query_positions(segment_ids.shape[1], start=self.end)
        end_after_write = self.end + appended_tokens
        kv_positions = self.slot_positions(end=end_after_write)
        kv_segment_ids = self.slot_segment_ids(end=end_after_write)
        return q_positions, kv_positions, q_segment_ids, kv_segment_ids, appended_tokens

    def __repr__(self):
        return layer_repr(self)


def update_slice(x: jax.Array, y: jax.Array, pos: int, update_axis: int):
    y = reshard(y.astype(x.dtype), jax.typeof(x).sharding.spec)
    return jax.lax.dynamic_update_slice_in_dim(x, y, pos, axis=update_axis)


def compute_segment_mask(segment_ids):
    """Compute once, reuse across all layers. Returns (B, 1, T, S) bias or None."""
    if segment_ids is None:
        return None

    # (B, T) valid token positions (segment_id != 0)
    valid_segment_ids = jnp.where(segment_ids != 0, 1, 0)

    # (B, T, T) same segment
    same_segment = jnp.equal(segment_ids[:, :, None], segment_ids[:, None, :])

    # (B, T, T) valid on both query and key axes
    valid = valid_segment_ids[:, :, None] & valid_segment_ids[:, None, :]

    mask = (same_segment & valid).astype(jnp.bool)
    return mask[:, None, :, :]  # (B, 1, T, T)


def make_attention_mask(
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
    q_positions: jax.Array,
    kv_positions: jax.Array,
    causal: bool,
    local_window_size=None,
):
    """Build an attention mask from explicit logical positions.

    Args:
        q_segment_ids: [B, T] binary or segment ids for query tokens.
        kv_segment_ids: [B, S] binary or segment ids for KV tokens.
        q_positions: [B, T] logical positions of query tokens.
        kv_positions: [B, S] logical positions of KV tokens.
        causal: Whether to enforce causal masking.
        local_window_size: Optional int or (left, right) tuple. For causal
            decode we only use the left window and restrict attention to keys
            within that many positions behind each query.
    """
    segment_mask = (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :])[
        :, None, :, :
    ]
    if causal:
        causal_mask = q_positions[:, None, :, None] >= kv_positions[:, None, None, :]
        if local_window_size is not None:
            if isinstance(local_window_size, int):
                left_window = local_window_size
            else:
                left_window = local_window_size[0]
            window_mask = (
                q_positions[:, None, :, None] - kv_positions[:, None, None, :]
            ) <= left_window
            causal_mask = causal_mask & window_mask
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
