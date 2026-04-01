import warnings
import jax
import jax.numpy as jnp
import dataclasses
from pathlib import Path
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


def init_uniform(scale=1.0):
    def kernel_init(key, shape, dtype):
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)

    return kernel_init


@dataclasses.dataclass
class EmbeddingConfig:
    dtype: jnp.dtype = jnp.bfloat16
    vocab_size: int = 50304
    d_emb: int = 768
    num_layers: int = 12
    weight_initializer: Callable = dataclasses.field(init=False)
    weight_logical_axes: Tuple[str, str] = ("vocab_out", "vocab_in")

    def __post_init__(self):
        self.weight_initializer = jax.nn.initializers.normal(stddev=1.0)


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

    wq_logical_axes: Tuple[str, str] = ("qkv_embed", "q_heads")
    wk_logical_axes: Tuple[str, str] = ("qkv_embed", "kv_heads")
    wv_logical_axes: Tuple[str, str] = ("qkv_embed", "kv_heads")
    wo_logical_axes: Tuple[str, str] = ("o_heads", "o_embed")

    def __post_init__(self):
        init = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wq_initializer = init
        self.wk_initializer = init
        self.wv_initializer = init
        self.wo_initializer = jax.nn.initializers.zeros


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

    wq_logical_axes: Tuple[str, str, str] = ("qkv_embed", "q_heads", "head_dim")
    wk_logical_axes: Tuple[str, str, str] = ("qkv_embed", "kv_heads", "head_dim")
    wv_logical_axes: Tuple[str, str, str] = ("qkv_embed", "kv_heads", "head_dim")
    wo_logical_axes: Tuple[str, str, str] = ("o_heads", "head_dim", "o_embed")

    def __post_init__(self):
        self.head_dim = self.d_emb // self.q_heads
        self.wq_initializer = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wk_initializer = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wv_initializer = init_uniform(scale=3**0.5 * self.d_emb**-0.5)
        self.wo_initializer = jax.nn.initializers.zeros


@dataclasses.dataclass
class LinearConfig:
    dtype: jnp.dtype = jnp.bfloat16
    in_features: int = 768
    out_features: int = 50304
    use_bias: bool = False
    weight_initializer: Callable = None
    weight_logical_axes: Tuple[str, str] = ("linear_in", "linear_out")


@dataclasses.dataclass
class MLPConfig:
    d_emb: int = 768
    dtype: jnp.dtype = jnp.bfloat16
    fc1: LinearConfig = dataclasses.field(init=False)
    fc2: LinearConfig = dataclasses.field(init=False)

    def __post_init__(self):
        self.fc1 = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb,
            out_features=self.d_emb * 4,
            weight_initializer=init_uniform(scale=3**0.5 * self.d_emb**-0.5),
            weight_logical_axes=("mlp_up_embed", "mlp_up_ffw"),
        )
        self.fc2 = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb * 4,
            out_features=self.d_emb,
            weight_initializer=jax.nn.initializers.zeros,
            weight_logical_axes=("mlp_down_ffw", "mlp_down_embed"),
        )


@dataclasses.dataclass
class ModelConfig:
    seqlen: int = 2048
    vocab_size: int = 50304
    d_emb: int = 768
    num_layers: int = 16
    q_heads: int = 8
    kv_heads: int = 4
    attn_type: str = "gqa"
    num_heads: Optional[Tuple[int, None]] = dataclasses.field(init=False)
    dtype: jnp.dtype = jnp.bfloat16

    embed: EmbeddingConfig = dataclasses.field(init=False)
    mlp: MLPConfig = dataclasses.field(init=False)
    lm_head: LinearConfig = dataclasses.field(init=False)

    # Attention Pattern: Can be global, local (SWA), or a combo.
    # Default is "L", meaning no SWA, and standard attention with RoPE is applied.
    # If using a combo of local-global, it is advised to use it in a ratio of 3:1
    # i.e. SSSL as first demosntrated by https://arxiv.org/abs/2501.18795
    window_pattern: str = dataclasses.field(metadata=dict(static=True), default="L")
    local_window_size: int = dataclasses.field(metadata=dict(static=True), default=None)

    # Again consistent with literature that we apply NoPE only to global attention, in
    # case it used with sliding window attention.
    nope_global_attn: bool = dataclasses.field(metadata=dict(static=True), default=True)

    if attn_type == "mha":
        attn: MultiHeadAttentionConfig = dataclasses.field(init=False)
    elif attn_type == "gqa":
        attn: GroupedQueryAttentionConfig = dataclasses.field(init=False)
    else:
        raise ValueError(
            f"Only these attention types are supported for now `['gqa', 'mha']`. Received = {attn_type}"
        )

    def __post_init__(self):
        if self.q_heads == self.kv_heads and self.attn_type == "gqa":
            raise ArithmeticError(
                "When the number of query heads equals the number of kv heads, JAX computes MHA not GQA!"
            )

        if self.local_window_size is not None and self.local_window_size >= self.seqlen:
            warnings.warn(
                "Window size for SWA cannot be greater than or equal to sequence length!"
            )
        elif self.window_pattern != "L" and self.local_window_size is None:
            warnings.warn(
                f"Looks like you want to use SWA, but did not pass window size. "
                f"Setting `window_size = seqlen // 4 = `{self.seqlen // 4}"
            )
            self.local_window_size = self.seqlen // 4

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
                num_layers=self.num_layers,
            )
        self.mlp = MLPConfig(dtype=self.dtype, d_emb=self.d_emb)
        self.lm_head = LinearConfig(
            dtype=self.dtype,
            in_features=self.d_emb,
            out_features=self.vocab_size,
            weight_logical_axes=("vocab_in", "vocab_out"),
            weight_initializer=jax.nn.initializers.normal(stddev=0.001),
        )


@dataclasses.dataclass
class ShardingRules:
    batch: AxisName = BATCH_AXIS_NAME
    sequence: AxisName = None
    act_embed: AxisName = None
    act_ffw: AxisName = None
    act_heads: AxisName = None
    head_dim: AxisName = None

    # Vocab projection/logit axes.
    vocab_in: AxisName = None
    vocab_out: AxisName = None

    # Attention parameter axes.
    qkv_embed: AxisName = None
    q_heads: AxisName = None
    kv_heads: AxisName = None
    o_heads: AxisName = None
    o_embed: AxisName = None

    # MLP parameter axes.
    mlp_up_embed: AxisName = None
    mlp_up_ffw: AxisName = None
    mlp_down_ffw: AxisName = None
    mlp_down_embed: AxisName = None

    # Reserved for future norm sharding.
    norm_in: AxisName = None
    norm_out: AxisName = None

    linear_in: AxisName = None
    linear_out: AxisName = None


@dataclasses.dataclass
class CheckpointConfig:
    # Checkpoint related
    max_checkpoints_to_keep: int = 5
    checkpoint_save_steps: int = 100
    last_checkpoint_step: int = 0
    # Directory where checkpoints will be saved
    save_ckpt_dir: Path | str = ""
    # Path to params subdirectory within a checkpoint from which weights will be loaded
    load_params_ckpt_path: Path | str = ""


@dataclasses.dataclass
class HyperParams:
    # Batch size related
    per_device_batch_size: int = 32
    desired_batch_size: int = 524288
    grad_accum_steps: Optional[float] = dataclasses.field(init=False)

    # Optimizer related
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    embedding_lr: float = 0.2
    unembedding_lr: float = 0.004
    other_peak_lr: float = 0.02
    b1: float = 0.8
    b2: float = 0.95
    weight_decay: float = 0.0
    cautious_weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    total_train_steps: int = 10000
    warmup_steps: int = int(min(300, 0.01 * total_train_steps))  # ~10% of total steps

    # For midtraining and SFT
    init_lr_frac: float = 0.2
    final_lr_frac: float = 0.0

    # Other
    es_patience: int = 500
    val_interval: int = 50


@jax_pytree_struct
class Config:
    seed: jax.Array = None
    mesh: Mesh = None
    rules: ShardingRules = dataclasses.field(default_factory=ShardingRules)
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    hparams: HyperParams = dataclasses.field(default_factory=HyperParams)
    ckpt_cfg: CheckpointConfig = dataclasses.field(default_factory=CheckpointConfig)
    data_dir: Path | str = "fineweb10B/"
