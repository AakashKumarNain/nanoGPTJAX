import math
import dataclasses

import jax
from utils import ParamSpec, ParamInitializer
from utils import layer_repr
from utils import jax_pytree_struct


def linear_init(fan_in, fan_out):
    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
    return jax.nn.initializers.normal(stddev=std)


def embed_init(std=1.0):
    return jax.nn.initializers.normal(stddev=std)


@jax_pytree_struct
class Linear(ParamInitializer):
    in_features: int = dataclasses.field(metadata=dict(static=True))
    out_features: int = dataclasses.field(metadata=dict(static=True))
    weight: jax.Array | ParamSpec
    bias: jax.Array | ParamSpec
    use_bias: bool = dataclasses.field(default=False, metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        weight = ParamSpec(
            shape=(cfg.in_features, cfg.out_features),
            dtype=cfg.dtype,
            logical_axes=cfg.weight_logical_axes,
            initializer=cfg.weight_initializer or linear_init(cfg.in_features, cfg.out_features),
        )
        if cfg.use_bias:
            bias = ParamSpec(
                shape=(cfg.out_features,),
                dtype=cfg.dtype,
                logical_axes=cfg.bias_logical_axes,
                initializer=cfg.bias_initializer or jax.nn.initializers.zeros,
            )
        else:
            bias = None
        return Linear(
            weight=weight,
            bias=bias,
            in_features=cfg.in_features,
            out_features=cfg.out_features,
            use_bias=cfg.use_bias,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return cls._init_fn(key, mesh, rules, cfg)

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class GroupedQueryAttention(ParamInitializer):
    wq: jax.Array | ParamSpec
    wk: jax.Array | ParamSpec
    wv: jax.Array | ParamSpec
    wo: jax.Array | ParamSpec
    d_emb: int = dataclasses.field(metadata=dict(static=True))
    q_heads: int = dataclasses.field(metadata=dict(static=True))
    kv_heads: int = dataclasses.field(metadata=dict(static=True))
    head_dim: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        wq = ParamSpec(
            shape=(cfg.d_emb, cfg.q_heads, cfg.head_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.wq_logical_axes,
            initializer=cfg.wq_initializer or linear_init(cfg.d_emb, cfg.head_dim),
        )
        wk = ParamSpec(
            shape=(cfg.d_emb, cfg.kv_heads, cfg.head_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.wk_logical_axes,
            initializer=cfg.wk_initializer or linear_init(cfg.d_emb, cfg.head_dim),
        )
        wv = ParamSpec(
            shape=(cfg.d_emb, cfg.kv_heads, cfg.head_dim),
            dtype=cfg.dtype,
            logical_axes=cfg.wv_logical_axes,
            initializer=cfg.wv_initializer or linear_init(cfg.d_emb, cfg.head_dim),
        )
        wo = ParamSpec(
            shape=(cfg.q_heads, cfg.head_dim, cfg.d_emb),
            dtype=cfg.dtype,
            logical_axes=cfg.wo_logical_axes,
            initializer=cfg.wo_initializer or linear_init(cfg.head_dim, cfg.d_emb),
        )

        return GroupedQueryAttention(
            d_emb=cfg.d_emb,
            q_heads=cfg.q_heads,
            kv_heads=cfg.kv_heads,
            head_dim=cfg.head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
        )

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return cls._init_fn(key, mesh, rules, cfg)

    def __repr__(self):
        return layer_repr(self)


@jax_pytree_struct
class Embedding(ParamInitializer):
    weight: jax.Array | ParamSpec
    vocab_size: int = dataclasses.field(metadata=dict(static=True))
    d_emb: int = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def param_specs(cls, cfg):
        weight = ParamSpec(
            shape=(cfg.vocab_size, cfg.d_emb),
            dtype=cfg.dtype,
            logical_axes=cfg.weight_logical_axes,
            initializer=cfg.weight_initializer or embed_init ,
        )
        return Embedding(vocab_size=cfg.vocab_size, d_emb=cfg.d_emb, weight=weight,)

    @classmethod
    def init(cls, key, mesh, rules, cfg):
        return cls._init_fn(key, mesh, rules, cfg)

    def __repr__(self):
        return layer_repr(self)