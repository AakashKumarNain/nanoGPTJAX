import jax
import jax.numpy as jnp
import dataclasses
from typing import Callable, Tuple, Optional
from jax.sharding import Mesh
from utils import jax_pytree_struct


AxisName = str | tuple[str, ...] | None
Axes = tuple[AxisName, ...]

# Expected physical mesh axis names:
# x - batch
# y - 1st of 2D tensor sharding
# z - 2nd of 2D tensor sharding
BATCH_AXIS_NAME = "x"
EXPERT_AXIS_NAME = "z"
TENSOR_ONLY_AXIS_NAME = "y"
ATTN_HEADS_AXIS_NAME = "y"
TENSOR_AXIS_NAME = ("y", "z")


def kernel_init(num_layers):
    std = 0.02 * (2 * num_layers) ** -0.5
    return jax.nn.initializers.normal(stddev=std)


@dataclasses.dataclass
class EmbeddingConfig:
    dtype: jnp.dtype = jnp.bfloat16
    vocab_size: int = 50304
    d_emb: int = 768
    num_layers: int = 12
    weight_initializer: Callable = dataclasses.field(init=False)
    weight_logical_axes: Tuple[str, str] = ("embed_in", "embed_out")

    def __post_init__(self):
        self.weight_initializer = kernel_init(self.num_layers)


@dataclasses.dataclass
class MultiHeadAttentionConfig:
    dtype: jnp.dtype = jnp.bfloat16
    d_in: int = 768
    d_out: int = 768
    num_heads: int = 12
    num_layers: int = 12

    wq_initializer: Callable = dataclasses.field(init=False)
    wk_initializer: Callable = dataclasses.field(init=False)
    wv_initializer: Callable = dataclasses.field(init=False)
    wo_initializer: Callable = dataclasses.field(init=False)

    wq_logical_axes: Tuple[str, str] = ("attn_wq_in", "attn_wq_out")
    wk_logical_axes: Tuple[str, str] = ("attn_wk_in", "attn_wk_out")
    wv_logical_axes: Tuple[str, str] = ("attn_wv_in", "attn_wv_out")
    wo_logical_axes: Tuple[str, str] = ("attn_wo_in", "attn_wo_out")

    def __post_init__(self):
        init = kernel_init(self.num_layers)
        self.wq_initializer = init
        self.wk_initializer = init
        self.wv_initializer = init
        self.wo_initializer = init


@dataclasses.dataclass
class GroupedQueryAttentionConfig:
    dtype: jnp.dtype = jnp.bfloat16
    d_emb: int = 768
    q_heads: int = 8
    kv_heads: int = 4
    num_layers: int = 12
    head_dim: int = dataclasses.field(init=False)

    wq_initializer: Callable = dataclasses.field(init=False)
    wk_initializer: Callable = dataclasses.field(init=False)
    wv_initializer: Callable = dataclasses.field(init=False)
    wo_initializer: Callable = dataclasses.field(init=False)

    wq_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_q_heads", "attn_head_dim")
    wk_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_kv_heads", "attn_head_dim")
    wv_logical_axes: Tuple[str, str, str] = ("attn_wqkv_in", "attn_kv_heads", "attn_head_dim")
    wo_logical_axes: Tuple[str, str, str] = ("attn_wo_in", "attn_head_dim", "attn_wo_out")

    def __post_init__(self):
        self.head_dim = self.d_emb // self.q_heads
        init = kernel_init(self.num_layers)
        self.wq_initializer = init
        self.wk_initializer = init
        self.wv_initializer = init
        self.wo_initializer = init


@dataclasses.dataclass
class LinearConfig:
    dtype: jnp.dtype = jnp.bfloat16
    in_features: int = 768
    out_features: int = 50304
    use_bias: bool = False
    num_layers: int = 12

    weight_initializer: Callable = dataclasses.field(init=False)
    weight_logical_axes: Tuple[str, str] = ("linear_in", "linear_out")

    def __post_init__(self):
        self.weight_initializer = kernel_init(self.num_layers)


@dataclasses.dataclass
class MLPConfig:
    d_emb: int = 768
    dtype: jnp.dtype = jnp.bfloat16
    num_layers: int = 12

    fc1: LinearConfig = dataclasses.field(init=False)
    fc2: LinearConfig = dataclasses.field(init=False)

    def __post_init__(self):
        self.fc1 = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb,
            out_features=self.d_emb * 4,
            num_layers=self.num_layers,
            weight_logical_axes=("mlp_fc1_in", "mlp_fc1_out"),
        )
        self.fc2 = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb * 4,
            out_features=self.d_emb,
            num_layers=self.num_layers,
            weight_logical_axes=("mlp_fc2_in", "mlp_fc2_out"),
        )

@dataclasses.dataclass
class ModelConfig:
    seqlen: int = 1024
    vocab_size: int = 50304
    d_emb: int = 768
    num_layers: int = 12
    q_heads: int = 6
    kv_heads: int = 6
    attn_type: str = "gqa"
    num_heads: Optional[Tuple[int, None]] = dataclasses.field(init=False)
    dtype: jnp.dtype = jnp.bfloat16

    embed: EmbeddingConfig = dataclasses.field(init=False)
    mlp: MLPConfig = dataclasses.field(init=False)
    lm_head: LinearConfig = dataclasses.field(init=False)

    if attn_type == "mha":
        attn: MultiHeadAttentionConfig = dataclasses.field(init=False)
    elif attn_type == "gqa":
        attn: GroupedQueryAttentionConfig = dataclasses.field(init=False)
    else:
        raise ValueError(f"Only these attention types are supported for now `['gqa', 'mha']`. Received = {attn_type}")

    def __post_init__(self):
        self.embed = EmbeddingConfig(
            dtype=self.dtype,
            vocab_size=self.vocab_size,
            d_emb=self.d_emb,
            num_layers=self.num_layers,
        )
        if self.attn_type == "mha":
            self.attn = MultiHeadAttentionConfig(
                dtype=self.dtype,
                d_in=self.d_emb,
                d_out=self.d_emb,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
            )
        elif self.attn_type == "gqa":
            self.attn = GroupedQueryAttentionConfig(
                dtype=self.dtype,
                d_emb=self.d_emb,
                q_heads=self.q_heads,
                kv_heads=self.kv_heads,
            )
        self.mlp = MLPConfig(
            dtype=self.dtype,
            d_emb=self.d_emb,
            num_layers=self.num_layers,
        )
        self.lm_head = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb,
            out_features=self.vocab_size,
            num_layers=self.num_layers,
            weight_logical_axes=("linear_in", "linear_out"),
        )


@dataclasses.dataclass
class ShardingRules:
    batch: AxisName = BATCH_AXIS_NAME
    sequence: AxisName = None
    act_embed: AxisName = None
    act_heads: AxisName = None

    embed_in: AxisName = None
    embed_out: AxisName = None

    attn_wqkv_in: AxisName = None
    attn_q_heads: AxisName = None
    attn_kv_heads: AxisName = None
    attn_head_dim: AxisName = None

    attn_wo_in: AxisName = None
    attn_wo_out: AxisName = None

    norm_in: AxisName = None
    norm_out: AxisName = None

    mlp_fc1_in: AxisName = None
    mlp_fc1_out: AxisName = None
    mlp_fc2_in: AxisName = None
    mlp_fc2_out: AxisName = None

    linear_in: AxisName = None
    linear_out: AxisName = None
    

@jax_pytree_struct
class Config:
    mesh: Mesh = None
    rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    per_device_batch_size: int = 32
    ckpt_dir: str = None
    load_ckpt_path: str = None
    data_dir: str = None
    
