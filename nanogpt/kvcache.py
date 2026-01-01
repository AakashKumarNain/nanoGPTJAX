"""
This implementation of KVCache has been adopted and modified from https://github.com/jax-ml/jax-llm-examples
There are certain aspects of this implementation which the beginners may find unintuitive, especially the
rolling buffer implementation.

TODO: Make it more intuitive to understand.
"""

import dataclasses
import jax
import jax.numpy as jnp
from jax.sharding import reshard
from utils import ParamSpec
from utils import ParamInitializer
from utils import jax_pytree_struct, layer_repr


@jax_pytree_struct
class KVCache(ParamInitializer):
    k: list[jax.Array]    # (batch_size, kv_heads, max_seq_len, head_dim)
    v: list[jax.Array]    # (batch_size, kv_heads, max_seq_len, head_dim)
    iter: jax.Array       # [] sequences are right-aligned for slice update performance
    starts: jax.Array     # [batch_size]  sequences are right-aligned, we need start indices
    time_axis: int = dataclasses.field(metadata=dict(static=True), default=2)
    size: int = dataclasses.field(metadata=dict(static=True), default=-1)

    @classmethod
    def param_specs(cls, batch_size, cfg):
        cache_spec = ParamSpec(
            shape=(batch_size, cfg.model.attn.kv_heads, cfg.model.seqlen, cfg.model.attn.head_dim),
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