import re
import math
import dataclasses

import jax
import jax.numpy as jnp
from jax import random
from jax import tree_util as jtu
from utils import ParamSpec
from utils import jax_pytree_struct


def int8_quant_init(key, shape, dtype=jnp.int8):
    return random.randint(key, shape, -128, 128, dtype=dtype)


def int8_scale_init(key, shape, dtype):
    return random.normal(key, shape, dtype=dtype) / math.sqrt(math.prod(shape)) / 127


def normalize_axes(axis, ndim: int) -> tuple[int, ...]:
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)

    normalized = []
    for ax in axis:
        ax = int(ax) % ndim
        normalized.append(ax)
    return tuple(sorted(set(normalized)))


def path_to_string(path):
    parts = []
    for key in path:
        if hasattr(key, "name"):
            parts.append(str(key.name))
        elif hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return ".".join(parts)


@jax_pytree_struct
class QArray:
    """Quantized tensor leaf with scale metadata.

    `quantized_value` stores the packed integer tensor. `scale` stores the
    dequantization scale after removing `reduction_axes` from the original
    tensor shape. For the weight-only PTQ path in this repo, those axes are the
    dimensions reduced by the model contractions, so scales usually correspond
    to output channels and can be applied after the dot/einsum.

    `scale_applied_to_output` and `scale_expand_dims` describe how the generic
    `einsum` wrapper should place the scale. They are execution hints, not part
    of the integer payload.
    """

    quantized_value: jax.Array | ParamSpec
    scale: jax.Array | ParamSpec
    reduction_axes: tuple[int, ...] = dataclasses.field(
        default=(), metadata=dict(static=True)
    )
    scale_applied_to_output: bool = dataclasses.field(
        default=True, metadata=dict(static=True)
    )
    scale_expand_dims: int | tuple[int, ...] = dataclasses.field(
        default=(), metadata=dict(static=True)
    )
    num_bits: int = dataclasses.field(default=8, metadata=dict(static=True))
    symmetric: bool = dataclasses.field(default=True, metadata=dict(static=True))
    dequantized_dtype: jnp.dtype = dataclasses.field(
        default=jnp.bfloat16, metadata=dict(static=True)
    )
    scale_dtype: jnp.dtype = dataclasses.field(
        default=jnp.bfloat16, metadata=dict(static=True)
    )

    @property
    def shape(self):
        return self.quantized_value.shape

    @property
    def ndim(self):
        return len(self.quantized_value.shape)


@dataclasses.dataclass(frozen=True)
class QuantizationRule:
    pattern: str
    axis: int | tuple[int, ...] | None = None
    scale_applied_to_output: bool = True
    scale_expand_dims: int | tuple[int, ...] = ()
    bits: int = 8
    symmetric: bool = True
    scale_dtype: jnp.dtype = jnp.bfloat16


def infer_weight_quant_axis(path: str, leaf, include_embeddings: bool = False):
    name = path.split(".")[-1]
    shape = leaf.shape
    ndim = len(shape)

    if name == "weight" and ndim == 2:
        if path.endswith("embed.weight"):
            return 1 if include_embeddings else None
        return 0

    if name in {"wq", "wk", "wv"} and ndim == 3:
        return 0

    if name == "wo" and ndim == 3:
        return (0, 1)

    return None


def quantize(
    x: jax.Array | ParamSpec,
    axis: int | tuple[int, ...],
    scale_dtype=jnp.bfloat16,
    zero_init: bool = False,
    scale_applied_to_output: bool = True,
    scale_expand_dims: int | tuple[int, ...] = (),
    bits: int = 8,
    symmetric: bool = True,
):
    """Quantize an array or spec over the given reduction axis/axes."""

    if isinstance(x, QArray):
        raise ValueError("Attempting to quantize an already quantized QArray.")
    if bits != 8:
        raise ValueError(
            "The minimal PTQ path currently supports only int8 quantization."
        )
    if not symmetric:
        raise ValueError(
            "The minimal PTQ path currently supports only symmetric quantization."
        )

    if isinstance(x, jax.Array):
        axis = normalize_axes(axis, x.ndim)
        scale_dtype = jnp.dtype(scale_dtype)
        x_for_quant = x.astype(scale_dtype)
        amax = jnp.max(jnp.abs(x_for_quant), axis=axis, keepdims=True)
        scale = jnp.maximum(amax / 127.0, jnp.finfo(scale_dtype).tiny).astype(
            scale_dtype
        )
        quantized_value = jnp.clip(jnp.rint(x_for_quant / scale), -127, 127).astype(
            jnp.int8
        )
        scale = scale.reshape(
            [dim for i, dim in enumerate(scale.shape) if i not in axis]
        )
        return QArray(
            quantized_value=quantized_value,
            scale=scale,
            reduction_axes=axis,
            scale_applied_to_output=scale_applied_to_output,
            scale_expand_dims=scale_expand_dims,
            num_bits=bits,
            symmetric=symmetric,
            dequantized_dtype=x.dtype,
            scale_dtype=scale_dtype,
        )

    if isinstance(x, ParamSpec):
        axis = normalize_axes(axis, len(x.shape))
        new_shape = tuple(dim for i, dim in enumerate(x.shape) if i not in axis)
        new_logical_axes = tuple(
            logical_axis
            for i, logical_axis in enumerate(x.logical_axes)
            if i not in axis
        )
        if zero_init:
            quant_init = jax.nn.initializers.zeros
            scale_init = jax.nn.initializers.ones
        else:
            quant_init = int8_quant_init
            scale_init = int8_scale_init
        quantized_value = dataclasses.replace(
            x,
            shape=x.shape,
            dtype=jnp.int8,
            initializer=quant_init,
        )
        scale = ParamSpec(
            shape=new_shape,
            dtype=jnp.dtype(scale_dtype),
            logical_axes=new_logical_axes,
            initializer=scale_init,
        )
        return QArray(
            quantized_value=quantized_value,
            scale=scale,
            reduction_axes=axis,
            scale_applied_to_output=scale_applied_to_output,
            scale_expand_dims=scale_expand_dims,
            num_bits=bits,
            symmetric=symmetric,
            dequantized_dtype=x.dtype,
            scale_dtype=jnp.dtype(scale_dtype),
        )
    raise ValueError(f"quantize got unexpected type: {type(x)}")


def dequantize_array(x: QArray, dtype=None) -> jax.Array:
    out_dtype = x.dequantized_dtype if dtype is None else dtype
    quantized_value = x.quantized_value.astype(out_dtype)
    scale = x.scale.astype(out_dtype)
    scale = scale.reshape(
        scale_shape_for_reduction_axes(x.scale, x.ndim, x.reduction_axes)
    )
    out = quantized_value * scale
    return out.astype(out_dtype)


def scale_shape_for_reduction_axes(
    scale: jax.Array | ParamSpec,
    original_ndim: int,
    reduction_axes: tuple[int, ...],
) -> tuple[int, ...]:
    full_scale_shape = []
    scale_idx = 0
    for dim in range(original_ndim):
        if dim in reduction_axes:
            full_scale_shape.append(1)
        else:
            full_scale_shape.append(scale.shape[scale_idx])
            scale_idx += 1
    return tuple(full_scale_shape)


def default_ptq_rules(include_embeddings: bool = False) -> list[QuantizationRule]:
    patterns = [
        r"(^|.*\.)weight$",
        r"(^|.*\.)wq$",
        r"(^|.*\.)wk$",
        r"(^|.*\.)wv$",
        r"(^|.*\.)wo$",
    ]
    if not include_embeddings:
        patterns[0] = r"^(?!(?:.*\.)?embed\.weight$)(?:^|.*\.)weight$"
    return [QuantizationRule(pattern=pattern) for pattern in patterns]


def quantize_params(
    params,
    rules: list[QuantizationRule] | None = None,
    *,
    include_embeddings: bool = False,
    return_stats: bool = True,
):
    if rules is None:
        rules = default_ptq_rules(include_embeddings=include_embeddings)

    compiled_rules = [(re.compile(rule.pattern), rule) for rule in rules]

    def maybe_quantize(path, leaf):
        if leaf is None or isinstance(leaf, QArray):
            return leaf
        if not isinstance(leaf, (jax.Array, ParamSpec)):
            return leaf

        path_str = path_to_string(path)
        for pattern, rule in compiled_rules:
            if not pattern.search(path_str):
                continue
            axis = rule.axis
            if axis is None:
                axis = infer_weight_quant_axis(
                    path_str, leaf, include_embeddings=include_embeddings
                )
            if axis is None:
                continue
            qleaf = quantize(
                leaf,
                axis=axis,
                scale_applied_to_output=rule.scale_applied_to_output,
                scale_expand_dims=rule.scale_expand_dims,
                bits=rule.bits,
                symmetric=rule.symmetric,
                scale_dtype=rule.scale_dtype,
            )
            return qleaf
        return leaf

    qparams = jtu.tree_map_with_path(
        maybe_quantize,
        params,
        is_leaf=lambda x: x is None or isinstance(x, QArray),
    )

    if not return_stats:
        return qparams

    stats = weight_size_reduction(params, qparams)
    return qparams, stats


def _leaf_nbytes(leaf) -> int:
    if leaf is None:
        return 0

    if isinstance(leaf, QArray):
        total = _leaf_nbytes(leaf.quantized_value) + _leaf_nbytes(leaf.scale)
        return total

    if isinstance(leaf, ParamSpec):
        return math.prod(leaf.shape) * jnp.dtype(leaf.dtype).itemsize

    if isinstance(leaf, jax.Array):
        return leaf.size * leaf.dtype.itemsize

    return 0


def tree_weight_nbytes(tree) -> int:
    leaves = jtu.tree_leaves(tree, is_leaf=lambda x: x is None or isinstance(x, QArray))
    return sum(_leaf_nbytes(leaf) for leaf in leaves)


def weight_size_reduction(reference_tree, quantized_tree) -> dict[str, float | int]:
    reference_nbytes = tree_weight_nbytes(reference_tree)
    quantized_nbytes = tree_weight_nbytes(quantized_tree)
    saved_nbytes = reference_nbytes - quantized_nbytes

    if reference_nbytes == 0:
        reduction_fraction = 0.0
        compression_ratio = 1.0
    else:
        reduction_fraction = saved_nbytes / reference_nbytes
        compression_ratio = (
            reference_nbytes / quantized_nbytes if quantized_nbytes else float("inf")
        )

    return {
        "reference_nbytes": reference_nbytes,
        "quantized_nbytes": quantized_nbytes,
        "saved_nbytes": saved_nbytes,
        "reduction_fraction": reduction_fraction,
        "reduction_percent": 100.0 * reduction_fraction,
        "compression_ratio": compression_ratio,
    }
