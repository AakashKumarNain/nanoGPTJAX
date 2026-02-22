from functools import partial

import jax
import jax.numpy as jnp

SOFTCAP_SCALE = 15.0
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_VOCAB_CHUNK_SIZE = 2048


def vocab_logsumexp(chunk_logits, vocab_chunk_size):
    vocab = chunk_logits.shape[-1]
    vchunk = min(vocab_chunk_size, vocab)
    num_chunks = (vocab + vchunk - 1) // vchunk
    last = vocab - 1
    valid_shape = (1,) * (chunk_logits.ndim - 1) + (vchunk,)

    max0 = jnp.full(chunk_logits.shape[:-1], -jnp.inf, jnp.float32)

    def max_body(i, max_vals):
        offset = i * vchunk
        vidx = offset + jnp.arange(vchunk)
        valid = vidx < vocab
        safe = jnp.minimum(vidx, last)

        logits_slice = jnp.take(chunk_logits, safe, axis=-1)
        logits_f32 = logits_slice.astype(jnp.float32)
        t = jnp.tanh(logits_f32 / SOFTCAP_SCALE)
        capped = SOFTCAP_SCALE * t

        valid_mask = valid.reshape(valid_shape)
        capped = jnp.where(valid_mask, capped, -jnp.inf)
        slice_max = jnp.max(capped, axis=-1)
        return jnp.maximum(max_vals, slice_max)

    max_vals = jax.lax.fori_loop(0, num_chunks, max_body, max0)

    def sum_body(i, sum_vals):
        offset = i * vchunk
        vidx = offset + jnp.arange(vchunk)
        valid = vidx < vocab
        safe = jnp.minimum(vidx, last)

        logits_slice = jnp.take(chunk_logits, safe, axis=-1)
        logits_f32 = logits_slice.astype(jnp.float32)
        t = jnp.tanh(logits_f32 / SOFTCAP_SCALE)
        capped = SOFTCAP_SCALE * t

        valid_mask = valid.reshape(valid_shape)
        exp = jnp.exp(capped - max_vals[..., None])
        exp = jnp.where(valid_mask, exp, 0.0)
        return sum_vals + jnp.sum(exp, axis=-1)

    sum0 = jnp.zeros_like(max_vals)
    sum_vals = jax.lax.fori_loop(0, num_chunks, sum_body, sum0)
    return jnp.log(sum_vals) + max_vals


def chunked_nll_sum(logits, labels, mask, chunk_size, vocab_chunk_size):
    seq_axis = labels.ndim - 1
    seq_len = labels.shape[seq_axis]
    if mask is None:
        total = jnp.array(labels.size, jnp.float32)
    else:
        total = jnp.sum(mask).astype(jnp.float32)
    chunk_size = min(chunk_size, seq_len)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    idxs = jnp.arange(num_chunks)
    last = seq_len - 1
    batch_ndim = labels.ndim - 1
    valid_shape = (1,) * batch_ndim + (chunk_size,)

    def body(carry, idx):
        offset = idx * chunk_size
        row_idxs = offset + jnp.arange(chunk_size)
        valid = row_idxs < seq_len
        safe_idxs = jnp.minimum(row_idxs, last)

        chunk_logits = jnp.take(logits, safe_idxs, axis=seq_axis)
        chunk_labels = jnp.take(labels, safe_idxs, axis=seq_axis)

        logsumexp = vocab_logsumexp(chunk_logits, vocab_chunk_size)
        label_raw = jnp.take_along_axis(
            chunk_logits, chunk_labels[..., None], axis=-1
        ).squeeze(-1)
        label_f32 = label_raw.astype(jnp.float32)
        t = jnp.tanh(label_f32 / SOFTCAP_SCALE)
        label_capped = SOFTCAP_SCALE * t

        nll = logsumexp - label_capped
        if mask is None:
            weight = valid.reshape(valid_shape).astype(jnp.float32)
        else:
            chunk_mask = jnp.take(mask, safe_idxs, axis=seq_axis).astype(jnp.float32)
            weight = chunk_mask * valid.reshape(valid_shape).astype(jnp.float32)
        nll = nll * weight
        return carry + jnp.sum(nll), None

    total_nll, _ = jax.lax.scan(body, jnp.array(0.0, jnp.float32), idxs)
    return total_nll, total


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def chunked_softmax_cross_entropy_with_integer_labels(
    logits, labels, mask=None, chunk_size=DEFAULT_CHUNK_SIZE
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    total_nll, total = chunked_nll_sum(
        logits, labels, mask, chunk_size, DEFAULT_VOCAB_CHUNK_SIZE
    )
    return jnp.where(total > 0, total_nll / total, 0.0)


def chunked_ce_fwd(logits, labels, mask, chunk_size):
    total_nll, total = chunked_nll_sum(
        logits, labels, mask, chunk_size, DEFAULT_VOCAB_CHUNK_SIZE
    )
    loss = jnp.where(total > 0, total_nll / total, 0.0)
    # Residuals should not include nondiff args.
    return loss, (logits, labels, mask, total)


def chunked_ce_bwd(chunk_size, res, g):
    logits, labels, mask, total = res
    seq_axis = labels.ndim - 1
    seq_len = labels.shape[seq_axis]
    chunk_size = min(chunk_size, seq_len)
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    padded_seq_len = num_chunks * chunk_size
    vocab = logits.shape[-1]
    vchunk = min(DEFAULT_VOCAB_CHUNK_SIZE, vocab)
    num_v = (vocab + vchunk - 1) // vchunk
    padded_vocab = num_v * vchunk
    last = seq_len - 1
    batch_ndim = labels.ndim - 1
    valid_seq_shape = (1,) * batch_ndim + (chunk_size,)
    valid_vocab_shape = (1,) * labels.ndim + (vchunk,)
    grad_shape = list(logits.shape)
    grad_shape[seq_axis] = padded_seq_len
    grad_shape[-1] = padded_vocab

    scale = jnp.where(total > 0, g.astype(jnp.float32) / total, 0.0).astype(jnp.float32)

    def seq_body(i, grad_out):
        offset = i * chunk_size
        row_idxs = offset + jnp.arange(chunk_size)
        valid_seq = row_idxs < seq_len
        safe_idxs = jnp.minimum(row_idxs, last)

        chunk_logits = jnp.take(logits, safe_idxs, axis=seq_axis)
        chunk_labels = jnp.take(labels, safe_idxs, axis=seq_axis)
        logsumexp = vocab_logsumexp(chunk_logits, vchunk)
        if mask is None:
            chunk_mask = None
        else:
            chunk_mask = jnp.take(mask, safe_idxs, axis=seq_axis).astype(jnp.float32)

        def vocab_body(j, grad_inner):
            v_offset = j * vchunk
            vidx = v_offset + jnp.arange(vchunk)
            valid_v = vidx < vocab
            safe_vidx = jnp.minimum(vidx, vocab - 1)

            logits_slice = jnp.take(chunk_logits, safe_vidx, axis=-1)
            logits_f32 = logits_slice.astype(jnp.float32)
            t = jnp.tanh(logits_f32 / SOFTCAP_SCALE)
            capped = SOFTCAP_SCALE * t

            valid_v_mask = valid_v.reshape(valid_vocab_shape)
            exp = jnp.exp(capped - logsumexp[..., None])
            exp = jnp.where(valid_v_mask, exp, 0.0)

            in_chunk = (chunk_labels >= v_offset) & (chunk_labels < v_offset + vchunk)
            labels_local = chunk_labels - v_offset
            softmax_flat = exp.reshape(-1, vchunk)
            labels_flat = labels_local.reshape(-1)
            in_flat = in_chunk.reshape(-1)
            token_idx = jnp.arange(labels_flat.shape[0])
            safe_labels = jnp.where(in_flat, labels_flat, 0)
            grad_flat = softmax_flat.at[token_idx, safe_labels].add(
                jnp.where(in_flat, -1.0, 0.0)
            )
            grad_capped = grad_flat.reshape(exp.shape)

            if chunk_mask is None:
                weight = valid_seq.reshape(valid_seq_shape)
            else:
                weight = chunk_mask * valid_seq.reshape(valid_seq_shape)
            grad_capped = grad_capped * weight[..., None]
            grad_capped = grad_capped * scale

            grad_raw = grad_capped * (1.0 - t * t)
            grad_chunk = grad_raw.astype(logits.dtype)

            start_indices = [0] * logits.ndim
            start_indices[seq_axis] = offset
            start_indices[-1] = v_offset
            return jax.lax.dynamic_update_slice(grad_inner, grad_chunk, start_indices)

        return jax.lax.fori_loop(0, num_v, vocab_body, grad_out)

    grad = jnp.zeros(grad_shape, dtype=logits.dtype)
    grad = jax.lax.fori_loop(0, num_chunks, seq_body, grad)
    if padded_seq_len != seq_len or padded_vocab != vocab:
        start = [0] * logits.ndim
        slice_sizes = list(logits.shape)
        grad = jax.lax.dynamic_slice(grad, start, slice_sizes)
    return (grad, None, None)


chunked_softmax_cross_entropy_with_integer_labels.defvjp(chunked_ce_fwd, chunked_ce_bwd)
