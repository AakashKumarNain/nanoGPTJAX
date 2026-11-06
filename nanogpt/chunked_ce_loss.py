"""Chunked linear cross entropy with XLA GEMMs.

Same API and maths as chunked_cross_entropy.py. sm_120 has no tcgen05 or WGMMA, so the GEMMs stay in XLA, the path
already tested on this GPU. Tokens are processed in chunks, limiting logits storage to [tokens_per_chunk, vocab_size].
The custom VJP is on the per-token losses; sum and mean are applied outside it.
- reduction 'none': the forward saves each token's logits maximum and log-sum; the backward recomputes the logits.
- reduction 'sum'/'mean': the forward also computes activation and weight gradients for a loss cotangent of 1.
  The backward reuses them for a common power-of-two scale; otherwise it recomputes.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P


@dataclass(frozen=True)
class ChunkConfig:
    """chunk_tokens: tokens per chunk. Logits have shape [chunk_tokens, vocab_size]; the default is not tuned.

    Configured chunks must have at least 16 rows; shards with fewer tokens use their actual row count.
    chunk_tokens=1 failed the 16-bit Optax parity tests,
    most likely because XLA lowers a single-row GEMM differently from the full projection.
    """

    chunk_tokens: int = 4096

    def __post_init__(self):
        if type(self.chunk_tokens) is not int or self.chunk_tokens < 16:
            raise ValueError("chunk_tokens must be an integer >= 16")


@dataclass(frozen=True)
class LossOptions:
    """Static options of one call, passed to the custom VJP."""

    soft_cap: float | None
    logits_dtype: str
    reduction: str
    chunk_tokens: int
    mesh: Mesh | AbstractMesh | None
    batch_axis: str


def matmul_precision(*arrays):
    # FP32 GEMMs run at full precision; the default may use TF32. 16-bit operands are unaffected.
    return (
        jax.lax.Precision.HIGHEST
        if any(a.dtype == jnp.float32 for a in arrays)
        else None
    )


def reduce_across_shards(collective, value, options):
    return value if options.mesh is None else collective(value, options.batch_axis)


def compute_chunked_loss_and_gradients(
    activations, weights, labels, active_mask, options, cotangent=None, saved_stats=None
):
    """Runs one shard over all chunks.

    Without cotangent: returns ((losses, logits_max, shifted_log_sum_exp), None).
    With cotangent: also returns the FP32 gradients (activation_grad, weight_grad) of sum(cotangent * losses).
    With cotangent and saved_stats: returns (None, (activation_grad, weight_grad)), recomputing only the logits.
    saved_stats contains (logits_max, shifted_log_sum_exp).
    Per-token outputs match labels; activation_grad matches activations; weight_grad is this shard's contribution.
    """
    token_shape, activation_shape, hidden_size = (
        labels.shape,
        activations.shape,
        weights.shape[0],
    )
    num_tokens = labels.size
    compute_statistics, compute_gradients = saved_stats is None, cotangent is not None

    # Inputs with fewer tokens than chunk_tokens form one chunk with as many rows as the input.
    tokens_per_chunk = max(1, min(options.chunk_tokens, num_tokens))
    active_mask = active_mask.reshape(num_tokens)
    # Masked activations are zeroed, so NaN padding cannot reach the loss or gradients.
    tokens = {
        "activations": jnp.where(
            active_mask[:, None], activations.reshape(num_tokens, hidden_size), 0
        ),
        "labels": labels.reshape(num_tokens),
        "active_mask": active_mask,
    }
    if compute_gradients:
        tokens["cotangent"] = (
            jnp.broadcast_to(cotangent, token_shape)
            .reshape(num_tokens)
            .astype(jnp.float32)
        )
    if not compute_statistics:
        tokens["logits_max"], tokens["shifted_log_sum_exp"] = (
            stat.reshape(num_tokens) for stat in saved_stats
        )
    # Padded tokens are inactive.
    padding_tokens = -num_tokens % tokens_per_chunk
    chunks = jax.tree.map(
        lambda array: jnp.pad(
            array, [(0, padding_tokens)] + [(0, 0)] * (array.ndim - 1)
        ).reshape(-1, tokens_per_chunk, *array.shape[1:]),
        tokens,
    )

    def process_chunk(weight_grad, chunk_tokens):
        chunk_activations = chunk_tokens["activations"]
        chunk_labels, chunk_mask = chunk_tokens["labels"], chunk_tokens["active_mask"]

        # 1. FP32 soft-capped logits.
        if options.logits_dtype == "input":
            logits = jnp.dot(
                chunk_activations,
                weights,
                precision=matmul_precision(chunk_activations),
            ).astype(jnp.float32)
        else:
            logits = jnp.dot(
                chunk_activations,
                weights,
                precision=matmul_precision(chunk_activations),
                preferred_element_type=jnp.float32,
            )
        soft_cap_derivative = None
        if options.soft_cap is not None:
            tanh_logits = jnp.tanh(logits / options.soft_cap)
            logits, soft_cap_derivative = (
                options.soft_cap * tanh_logits,
                (1.0 - tanh_logits) * (1.0 + tanh_logits),
            )

        # 2. Softmax statistics and losses, or the saved statistics.
        if compute_statistics:
            # Max and log-sum stay separate: a fused log-sum-exp loses softmax precision at large offsets.
            logits_max = jnp.max(logits, axis=1)
            shifted_log_sum_exp = jnp.log(
                jnp.sum(jnp.exp(logits - logits_max[:, None]), axis=1)
            )
            target_logits = jnp.take_along_axis(
                logits, jnp.maximum(chunk_labels, 0)[:, None], axis=1
            )[:, 0]
            # Same order as Optax, logsumexp - target_logits. Invalid active labels give NaN, not silent masking.
            losses = jnp.where(
                chunk_labels >= 0,
                (logits_max + shifted_log_sum_exp) - target_logits,
                jnp.nan,
            )
            stats = (
                jnp.where(chunk_mask, losses, 0.0),
                logits_max,
                shifted_log_sum_exp,
            )
        else:
            logits_max, shifted_log_sum_exp = (
                chunk_tokens["logits_max"],
                chunk_tokens["shifted_log_sum_exp"],
            )
            stats = None
        if not compute_gradients:
            return weight_grad, (stats, None)

        # 3. logits_grad = d(loss)/d(logits before capping), rounded to logits_grad_dtype.
        # One FP32 factor per token: its cotangent, NaN for an invalid active label, and 0 when masked.
        token_gradient_scale = jnp.where(
            chunk_mask,
            jnp.where(chunk_labels >= 0, chunk_tokens["cotangent"], jnp.nan),
            0.0,
        )
        logits_grad_dtype = (
            activations.dtype
            if options.logits_dtype == "input"
            else jnp.dtype(jnp.float32)
        )
        probabilities = jnp.exp(
            (logits - logits_max[:, None]) - shifted_log_sum_exp[:, None]
        )
        columns = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        logits_grad = (
            jnp.where(
                columns == chunk_labels[:, None], probabilities - 1.0, probabilities
            )
            * token_gradient_scale[:, None]
        )
        if soft_cap_derivative is not None:
            logits_grad = logits_grad * soft_cap_derivative
        logits_grad = logits_grad.astype(logits_grad_dtype)

        # 4. Activation and weight gradients, accumulated in FP32.
        precision = matmul_precision(chunk_activations, logits_grad)
        chunk_activation_grad = jnp.dot(
            logits_grad,
            weights.T,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        chunk_weight_grad = jnp.dot(
            chunk_activations.T,
            logits_grad,
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return weight_grad + chunk_weight_grad, (stats, chunk_activation_grad)

    initial_weight_grad = (
        jnp.zeros(weights.shape, jnp.float32) if compute_gradients else None
    )
    weight_grad, (stats, activation_grad) = jax.lax.scan(
        process_chunk, initial_weight_grad, chunks
    )
    if stats is not None:
        stats = tuple(
            stat.reshape(-1)[:num_tokens].reshape(token_shape) for stat in stats
        )
    if not compute_gradients:
        return stats, None
    return stats, (
        activation_grad.reshape(-1, hidden_size)[:num_tokens].reshape(activation_shape),
        weight_grad,
    )


def reduction_cotangent(active_mask, options):
    """The cotangent that sum or mean gives each token for a loss cotangent of 1.

    It repeats the reduction in linear_cross_entropy, so the backward compares bit-identical values:
    1 for sum, 1 / max(count, 1) for mean with the global FP32 token count.
    """
    if options.reduction == "sum":
        return jnp.ones((), jnp.float32)
    count = reduce_across_shards(
        jax.lax.psum, jnp.sum(active_mask, dtype=jnp.float32), options
    )
    return 1.0 / jnp.maximum(count, 1.0)


def map_over_batch_shards(function, options, in_specs, out_specs):
    """Run on each batch shard. The VJP handles collectives, so variance checking is off."""
    if options.mesh is None:
        return function
    return jax.shard_map(
        function,
        mesh=options.mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def compute_token_losses(activations, weights, labels, active_mask, options):
    # Without differentiation, only the forward runs.
    batch_spec = P(options.batch_axis)

    def forward(activations, weights, labels, active_mask):
        return compute_chunked_loss_and_gradients(
            activations, weights, labels, active_mask, options
        )[0][0]

    return map_over_batch_shards(
        forward, options, (batch_spec, P(), batch_spec, batch_spec), batch_spec
    )(activations, weights, labels, active_mask)


def token_losses_forward(activations, weights, labels, active_mask, options):
    batch_spec = P(options.batch_axis)
    if options.reduction == "none":

        def forward(activations, weights, labels, active_mask):
            return compute_chunked_loss_and_gradients(
                activations, weights, labels, active_mask, options
            )[0]

        num_outputs = 3
    else:

        def forward(activations, weights, labels, active_mask):
            cotangent = reduction_cotangent(active_mask, options)
            stats, (activation_grad, weight_grad) = compute_chunked_loss_and_gradients(
                activations, weights, labels, active_mask, options, cotangent=cotangent
            )
            # A leading shard axis keeps each shard's own weight_grad.
            return (*stats, activation_grad, weight_grad[None])

        num_outputs = 5
    losses, *saved_values = map_over_batch_shards(
        forward,
        options,
        (batch_spec, P(), batch_spec, batch_spec),
        (batch_spec,) * num_outputs,
    )(activations, weights, labels, active_mask)
    return losses, (activations, weights, labels, active_mask, *saved_values)


def token_losses_backward(options, residuals, token_loss_cotangent):
    def recompute(
        activations,
        weights,
        labels,
        active_mask,
        logits_max,
        shifted_log_sum_exp,
        token_loss_cotangent,
    ):
        return compute_chunked_loss_and_gradients(
            activations,
            weights,
            labels,
            active_mask,
            options,
            cotangent=token_loss_cotangent,
            saved_stats=(logits_max, shifted_log_sum_exp),
        )[1]

    def backward(
        activations,
        weights,
        labels,
        active_mask,
        logits_max,
        shifted_log_sum_exp,
        *rest,
    ):
        if options.reduction == "none":
            (token_loss_cotangent,) = rest
            activation_grad, weight_grad = recompute(
                activations,
                weights,
                labels,
                active_mask,
                logits_max,
                shifted_log_sum_exp,
                token_loss_cotangent,
            )
        else:
            saved_activation_grad, saved_weight_grad, token_loss_cotangent = rest
            # Reuse gradients when all active cotangents share a power-of-two scale. This commutes with rounding
            # except where a value leaves the normal floating-point range.
            mantissa, exponent = jnp.frexp(token_loss_cotangent.astype(jnp.float32))
            scale_mantissa, scale_exponent = jnp.frexp(
                reduction_cotangent(active_mask, options)
            )
            exponent_shift = exponent - scale_exponent
            max_int32 = jnp.iinfo(jnp.int32).max
            # All shards must take the same branch, so the test reduces across shards.
            mantissa_mismatches = jnp.sum(
                active_mask & (mantissa != scale_mantissa), dtype=jnp.int32
            )
            mantissa_mismatches = reduce_across_shards(
                jax.lax.psum, mantissa_mismatches, options
            )
            min_exponent_shift = reduce_across_shards(
                jax.lax.pmin,
                jnp.min(jnp.where(active_mask, exponent_shift, max_int32)),
                options,
            )
            max_exponent_shift = reduce_across_shards(
                jax.lax.pmax,
                jnp.max(jnp.where(active_mask, exponent_shift, -max_int32)),
                options,
            )
            # With no active tokens, min_exponent_shift > max_exponent_shift and the saved gradients are zero.
            can_reuse_saved_gradients = (mantissa_mismatches == 0) & (
                min_exponent_shift >= max_exponent_shift
            )
            exponent_shift = jnp.where(
                min_exponent_shift == max_exponent_shift, min_exponent_shift, 0
            )
            activation_grad, weight_grad = jax.lax.cond(
                can_reuse_saved_gradients,
                lambda: (
                    jnp.ldexp(saved_activation_grad, exponent_shift),
                    jnp.ldexp(saved_weight_grad[0], exponent_shift),
                ),
                lambda: recompute(
                    activations,
                    weights,
                    labels,
                    active_mask,
                    logits_max,
                    shifted_log_sum_exp,
                    token_loss_cotangent,
                ),
            )
        # Replicated weights need every shard's part: sum in FP32, outside the cond, then cast.
        weight_grad = reduce_across_shards(jax.lax.psum, weight_grad, options)
        return activation_grad.astype(activations.dtype), weight_grad.astype(
            weights.dtype
        )

    batch_spec = P(options.batch_axis)
    inputs = (*residuals, token_loss_cotangent)
    # Weights (input 1) are replicated; all other inputs, including stacked weight gradients, split on axis 0.
    in_specs = tuple(P() if index == 1 else batch_spec for index in range(len(inputs)))
    activation_grad, weight_grad = map_over_batch_shards(
        backward, options, in_specs, (batch_spec, P())
    )(*inputs)
    return activation_grad, weight_grad, None, None


compute_token_losses.defvjp(token_losses_forward, token_losses_backward)


def linear_cross_entropy(
    x: jax.Array,
    w: jax.Array,
    labels: jax.Array,
    *,
    mask: jax.Array | None = None,
    ignore_index: int | None = -100,
    reduction: Literal["none", "sum", "mean"] = "mean",
    soft_cap: float | None = 15.0,
    logits_dtype: Literal["float32", "input"] = "float32",
    mesh: Mesh | AbstractMesh | None = None,
    batch_axis: str = "x",
    chunk_tokens: int = 4096,
) -> jax.Array:
    """Compute cross entropy of ``x @ w``, one chunk of tokens at a time.

    Args:
        x: Activations [..., H], with at least one token dimension.
        w: Weights [H, V], same dtype as x: float32, bfloat16, or float16.
        labels: Signed integer targets [...]. Active targets must be in [0, V).
        mask: Boolean array matching labels exactly; True includes a token.
        ignore_index: Additional ignored label value, or None to disable it.
        reduction: 'none' returns [...]; 'sum' and 'mean' return scalars.
            Mean divides by the active token count. Empty/all-masked inputs
            produce zero loss and gradients for every reduction.
        soft_cap: Positive finite scale for ``scale * tanh(logits / scale)``.
            None disables capping. nanoGPTJAX uses 15.0.
        logits_dtype: 'input' rounds logits and their cotangents to x.dtype,
            matching nanoGPTJAX's projection. 'float32' keeps FP32 logits.
        mesh: Optional one-dimensional data-parallel mesh. Shards x, labels and
            mask on their first dimension; w must be replicated. Reductions use
            the global token count, and weight gradients are summed in FP32.
        batch_axis: Name of the mesh axis, 'x' in nanoGPTJAX. The first dimension
            must be divisible by the number of devices. Close over mesh in jit.
        config: Static chunk configuration. Close over it when using jax.jit.

    Losses and gradient accumulation use float32. Gradients have input dtypes.
    Labels are already aligned: callers must apply the causal shift themselves.
    Supports jit and first-order reverse-mode autodiff. Higher-order and
    forward-mode AD are not implemented.
    For 'sum' and 'mean', the VJP forward saves FP32 activation gradients [..., H] and weight gradients [H, V]
    per device. The backward uses them when the loss cotangent is a power of two
    (for example 1, or 1 / grad_accum_steps with a power-of-two step count); otherwise it recomputes the logits.
    """
    activations, weights = x, w
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    if ignore_index is not None and type(ignore_index) is not int:
        raise TypeError("ignore_index must be an integer or None")
    if soft_cap is not None:
        if (
            type(soft_cap) not in (int, float)
            or not math.isfinite(soft_cap)
            or soft_cap <= 0
        ):
            raise ValueError("soft_cap must be a positive finite Python scalar or None")
        soft_cap = float(soft_cap)
    if logits_dtype not in ("float32", "input"):
        raise ValueError("logits_dtype must be 'float32' or 'input'")
    activations, weights, labels = (
        jnp.asarray(activations),
        jnp.asarray(weights),
        jnp.asarray(labels),
    )
    if (
        activations.ndim < 2
        or weights.ndim != 2
        or activations.shape[-1] != weights.shape[0]
    ):
        raise ValueError(
            "expected activations [..., H] and weights [H, V] with matching H"
        )
    if weights.shape[0] == 0 or weights.shape[1] == 0:
        raise ValueError("hidden size and vocabulary size must be positive")
    if labels.shape != activations.shape[:-1]:
        raise ValueError("labels must match activations.shape[:-1]")
    if mesh is not None:
        if not isinstance(mesh, (Mesh, AbstractMesh)):
            raise TypeError("mesh must be a JAX Mesh or AbstractMesh")
        if mesh.axis_names != (batch_axis,):
            raise ValueError("mesh must have exactly one axis named batch_axis")
        if activations.shape[0] % mesh.shape[batch_axis]:
            raise ValueError(
                "the first input dimension must be divisible by the mesh size"
            )
    supported_dtypes = (
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float16),
    )
    if activations.dtype not in supported_dtypes or weights.dtype != activations.dtype:
        raise TypeError(
            "activations and weights must have the same float32, bfloat16, or float16 dtype"
        )
    if not jnp.issubdtype(labels.dtype, jnp.signedinteger):
        raise TypeError("labels must have a signed integer dtype")
    # Small signed types must not narrow the vocabulary bound during comparison.
    labels = labels.astype(jnp.promote_types(labels.dtype, jnp.int32))
    if mask is None:
        active_mask = jnp.ones(labels.shape, jnp.bool_)
    else:
        active_mask = jnp.asarray(mask)
        if active_mask.shape != labels.shape:
            raise ValueError("mask must have exactly the labels shape")
        if active_mask.dtype != jnp.bool_:
            raise TypeError("mask must have Boolean dtype")
    if ignore_index is not None:
        active_mask = active_mask & (labels != ignore_index)
    # Canonicalize invalid labels to -1 before narrowing, so int64 labels cannot wrap.
    labels = jnp.where((labels >= 0) & (labels < weights.shape[1]), labels, -1).astype(
        jnp.int32
    )

    options = LossOptions(
        soft_cap, logits_dtype, reduction, chunk_tokens, mesh, batch_axis
    )
    losses = compute_token_losses(activations, weights, labels, active_mask, options)
    if reduction == "none":
        return losses
    # Global reductions over all shards; reduction_cotangent repeats them for the fast path.
    total = jnp.sum(losses, dtype=jnp.float32)
    if reduction == "sum":
        return total
    return total / jnp.maximum(jnp.sum(active_mask, dtype=jnp.float32), 1.0)
