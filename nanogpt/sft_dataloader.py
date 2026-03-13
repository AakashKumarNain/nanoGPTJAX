import json
import grain
import tiktoken
import argparse
import numpy as np
from pathlib import Path
from functools import partial

import pyarrow as pa
import pyarrow.parquet as pq

from tasks.mmlu import MMLU

# from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk


ROLE_MAP = {
    "system": ("system_start", "system_end"),
    "user": ("user_start", "user_end"),
    "assistant": ("assistant_start", "assistant_end"),
    "tool": ("tool_start", "tool_end"),
}


def prepare_train_batch(batch):
    """x/y split, align segment_ids, positions, and completion_mask to y."""
    ids = batch["input_ids"]
    seg = batch.get("input_ids_segment_ids")
    pos = batch.get("input_ids_positions")
    mask = batch.get("completion_mask")

    out = {"x": ids[:, :-1], "y": ids[:, 1:]}
    if seg is not None:
        out["segment_ids"] = seg[:, :-1]
    if pos is not None:
        out["positions"] = pos[:, :-1]
    if mask is not None:
        out["completion_mask"] = mask[:, 1:]
    return out


def prepare_train_accum_batch(batch, grad_accum_steps):
    """Same as prepare_train_batch but reshapes for gradient accumulation."""
    ids = batch["input_ids"]
    bsz = ids.shape[0] // grad_accum_steps

    def reshape(arr):
        if arr is None:
            return None
        return arr.reshape(grad_accum_steps, bsz, *arr.shape[1:])

    ids = reshape(ids)
    seg = reshape(batch.get("input_ids_segment_ids"))
    pos = reshape(batch.get("input_ids_positions"))
    mask = reshape(batch.get("completion_mask"))

    out = {"x": ids[:, :, :-1], "y": ids[:, :, 1:]}
    if seg is not None:
        out["segment_ids"] = seg[:, :, :-1]
    if pos is not None:
        out["positions"] = pos[:, :, :-1]
    if mask is not None:
        out["completion_mask"] = mask[:, :, 1:]
    return out


def truncate_to_seqlen(sample, max_len: int):
    ids = sample["input_ids"]
    mask = sample["completion_mask"]
    # Truncate both consistently
    if ids.shape[0] > max_len:
        ids = ids[:max_len]
        mask = mask[:max_len]
    sample["input_ids"] = ids
    sample["completion_mask"] = mask
    return sample


def has_completion_tokens(example):
    """
    Filter out chunks where truncation left only masked (prompt) tokens.
    Such chunks would cause CrossEntropyLoss(reduction='mean') to return
    nan due to 0/0 when every label is the ignore index.
    """
    return bool(example["completion_mask"].any())


def build_tokenizer():
    """Build a GPT-2 tokenizer extended with custom chat tokens."""
    user_start = "<|user_start|>"
    user_end = "<|user_end|>"
    assistant_start = "<|assistant_start|>"
    assistant_end = "<|assistant_end|>"
    system_start = "<|system_start|>"
    system_end = "<|system_end|>"
    tool_start = "<|tool_start|>"
    tool_end = "<|tool_end|>"
    pad_token = "<|pad|>"

    custom_tokens = [
        pad_token,
        user_start,
        user_end,
        assistant_start,
        assistant_end,
        system_start,
        system_end,
        tool_start,
        tool_end,
    ]

    base = tiktoken.get_encoding("gpt2")
    custom_token_ids = {tok: base.n_vocab + i for i, tok in enumerate(custom_tokens)}

    tokenizer = tiktoken.Encoding(
        name="gpt2_with_custom_tokens",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **custom_token_ids},
    )

    bos_id = tokenizer.eot_token
    bos = tokenizer.decode([bos_id])

    return {
        "tokenizer": tokenizer,
        "bos_id": bos_id,
        "bos": bos,
        "user_start": user_start,
        "user_end": user_end,
        "assistant_start": assistant_start,
        "assistant_end": assistant_end,
        "system_start": system_start,
        "system_end": system_end,
        "tool_start": tool_start,
        "tool_end": tool_end,
        "pad_token": pad_token,
        "pad_id": custom_token_ids[pad_token],
        "assistant_start_id": custom_token_ids[assistant_start],
        "assistant_end_id": custom_token_ids[assistant_end],
        "custom_token_ids": custom_token_ids,
        "vocab_size": tokenizer.n_vocab,
    }


def encode_mask_into_ids(input_ids, completion_mask):
    """Fuse a boolean completion_mask into the sign of input_ids."""
    return np.where(completion_mask, input_ids, -(input_ids + 1)).astype(np.int32)


def decode_mask_from_ids(batch):
    """Recover (unsigned ids, bool mask) from sign-encoded ids."""
    ids = batch["input_ids"]
    mask = ids >= 0
    batch["input_ids"] = np.where(mask, ids, -(ids + 1)).astype(np.int32)
    batch["completion_mask"] = mask
    return batch


def format_conversation(example, tok_info, check_assistant_role=True):
    """Format a single or multi-turn chat example into a string ready
        for tokenization.

    Args:
        example (dict): A conversation packed in a dict with some roles
        tok_info: A dictionary containing information about the tokenizer
        check_assistant_role (bool): Check if there are any completion
            message in the formatted conversation or not. Defaults to True

    """
    messages = example.get("messages", [])
    if not messages:
        return None

    parts = [tok_info["bos"]]

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if not content or role not in ROLE_MAP:
            continue

        start_key, end_key = ROLE_MAP[role]
        parts.append(f"{tok_info[start_key]}{content}{tok_info[end_key]}\n")

    if check_assistant_role:
        # This is only applicable during training where we are formatting
        # conversations and masking out non-completion tokens. There might be
        # cases where we do not have the completion tokens in the formatted
        # text (too lengthy conv, bad sample, etc). In those case, it is better
        # to skip those samples.
        roles = {m.get("role") for m in messages}
        if "assistant" not in roles:
            return None

    return {"text": "".join(parts)}


def tokenize_dataset(ds, tok_info, threshold_percentile=None, num_threads=32):
    tokenizer = tok_info["tokenizer"]
    ast_start_id = tok_info["assistant_start_id"]
    ast_end_id = tok_info["assistant_end_id"]

    # Format conversations
    formatted = []
    for example in ds:
        result = format_conversation(example, tok_info)
        if result is not None:
            formatted.append(result["text"])

    # Batch tokenize
    all_tokens = tokenizer.encode_batch(
        formatted,
        num_threads=num_threads,
        allowed_special="all",
    )

    # Stats
    lengths = np.array([len(t) for t in all_tokens])
    log_lengths = np.log1p(lengths)
    q1, q3 = np.percentile(log_lengths, [25, 75])
    iqr = q3 - q1
    fence = np.expm1(q3 + 3.0 * iqr)

    print(f"Total formatted:   {len(lengths)}")
    print(f"Median:            {np.expm1(np.median(log_lengths)):.0f} tokens")
    print(f"P95:               {np.expm1(np.percentile(log_lengths, 95)):.0f} tokens")
    print(f"P99:               {np.expm1(np.percentile(log_lengths, 99)):.0f} tokens")
    print(f"log-IQR fence:     {fence:.0f} tokens")

    # (Optional) Filter
    if threshold_percentile is not None:
        threshold = int(np.percentile(lengths, threshold_percentile))
        print(f"Threshold (P{threshold_percentile}):   {threshold} tokens")
        keep_mask = lengths <= threshold
    else:
        keep_mask = np.ones(len(lengths), dtype=bool)

    num_kept = int(keep_mask.sum())
    print(
        f"Tokens kept:  {num_kept} / {len(lengths)} ({100 * num_kept / len(lengths):.2f}%)"
    )

    # Compute completion mask + sign-bit encode
    encoded = []
    num_no_completion = 0  # some completions may get truncated after filtering

    for tokens, keep in zip(all_tokens, keep_mask):
        if not keep:
            continue

        tokens_arr = np.array(tokens, dtype=np.int32)
        starts = np.where(tokens_arr == ast_start_id)[0]
        ends = np.where(tokens_arr == ast_end_id)[0]
        completion_mask = np.zeros(len(tokens_arr), dtype=bool)
        num_pairs = min(len(starts), len(ends))

        for curr_start, curr_end in zip(starts[:num_pairs], ends[:num_pairs]):
            completion_mask[curr_start + 1 : curr_end + 1] = True

        if len(starts) > len(ends) and len(starts) > 0:
            completion_mask[starts[-1] + 1 :] = True

        if not completion_mask.any():  # drop prompt-only samples
            num_no_completion += 1
            continue

        encoded.append(encode_mask_into_ids(tokens_arr, completion_mask))

    num_kept = len(encoded)
    print(
        f"Tokens kept: {num_kept} / {len(lengths)} ({100 * num_kept / len(lengths):.2f}%)"
    )
    print(f"Dropped (length): {int((~keep_mask).sum())}")
    print(f"Dropped (no completion): {num_no_completion}")
    return encoded


def save_to_parquet(
    encoded, output_dir, task_name, split="train", rows_per_shard=64_000
):
    out = Path(output_dir) / task_name
    out.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([pa.field("input_ids", pa.list_(pa.int32()))])
    shard_idx = 0
    buf = []
    num_written = 0

    def flush(arrays, idx):
        offsets = np.zeros(len(arrays) + 1, dtype=np.int32)

        for idx, array in enumerate(arrays):
            offsets[idx + 1] = offsets[idx] + len(array)

        flat = np.concatenate(arrays)
        pa_col = pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()), pa.array(flat, type=pa.int32())
        )

        path = out / f"{split}_{shard_idx:05d}.parquet"
        pq.write_table(
            pa.table({"input_ids": pa_col}, schema=schema), path, compression="zstd"
        )
        print(f"Shard {shard_idx:05d} containing {len(arrays)} rows → {path}")

    for arr in encoded:
        buf.append(arr)
        num_written += 1
        if len(buf) >= rows_per_shard:
            flush(buf, shard_idx)
            buf = []
            shard_idx += 1

    if buf:
        flush(buf, shard_idx)
        shard_idx += 1

    meta = {
        "task": task_name,
        "split": split,
        "n_rows": num_written,
        "n_shards": shard_idx,
        "rows_per_shard": rows_per_shard,
    }
    with open(out / f"{split}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Written {num_written} rows in {shard_idx} shard -> {out}")


def make_grain_shard_loader(
    data_dir: str,
    split: str,
    pad_id: int,
    batch_size: int,
    sequence_length: int,
    grad_accum_steps: int = 1,
    data_sharding=None,
    # Packing controls
    num_packing_bins: int = 64,
    max_sequences_per_bin: int = 16,
    # Laziness / performance controls (thread-only)
    cycle_length: int = 8,
    iter_buffer_size: int = 64,
    num_make_iter_threads: int = 2,
    make_iter_buffer_size: int = 4,
    # Multithreading
    multi_threading: bool = True,
    prefetch_threads: int = 8,
    prefetch_buffer_size: int = 256,
    # Device put buffers
    cpu_buffer_size: int = 16,
    device_buffer_size: int = 4,
    # Shuffle
    shuffle=False,
    shuffle_seed: int = 1234,
):
    packed_len = sequence_length + 1
    shard_paths = list(Path(data_dir).glob(f"**/*{split}*.parquet"))
    print(f"Number of {split} files found: {len(shard_paths)}")
    paths_ds = grain.MapDataset.source(shard_paths)
    per_file = paths_ds.map(lambda p: grain.experimental.ParquetIterDataset(p))

    ds = grain.experimental.InterleaveIterDataset(
        per_file,
        cycle_length=min(cycle_length, len(shard_paths)),
        num_make_iter_threads=num_make_iter_threads,
        make_iter_buffer_size=make_iter_buffer_size,
        iter_buffer_size=iter_buffer_size,
    )
    if shuffle:
        ds = ds.shuffle(shuffle_seed)

    ds = ds.map(decode_mask_from_ids)
    ds = ds.map(partial(truncate_to_seqlen, max_len=packed_len))
    ds = ds.filter(has_completion_tokens)

    length_struct = {"input_ids": packed_len, "completion_mask": packed_len}
    padding_struct = {"input_ids": pad_id, "completion_mask": 0}

    ds = grain.experimental.BestFitPackIterDataset(
        parent=ds,
        length_struct=length_struct,
        num_packing_bins=num_packing_bins,
        max_sequences_per_bin=max_sequences_per_bin,
        padding_struct=padding_struct,
    )

    total_batch_size = (
        grad_accum_steps * batch_size if grad_accum_steps > 1 else batch_size
    )

    if multi_threading:
        ds = grain.experimental.multithread_prefetch(
            ds, num_threads=prefetch_threads, buffer_size=prefetch_buffer_size
        )

    ds = ds.batch(total_batch_size, drop_remainder=True)

    if grad_accum_steps > 1:
        ds = ds.map(
            partial(prepare_train_accum_batch, grad_accum_steps=grad_accum_steps)
        )
    else:
        ds = ds.map(prepare_train_batch)

    if data_sharding is not None:
        ds = grain.experimental.device_put(
            ds,
            device=data_sharding,
            cpu_buffer_size=cpu_buffer_size,
            device_buffer_size=device_buffer_size,
        )
    return ds


def main(args):
    tok_info = build_tokenizer()
    train_tasks = {
        "smoltalk": SmolTalk(split="train"),
        "mmlu": MMLU(subset="auxiliary_train", split="train"),
        "gsm8k": GSM8K(subset="main", split="train"),
        # "simplespelling": SimpleSpelling(size=200000, split="train", base_dir=data_dir),
        # "spellingbee":   SpellingBee(size=80000, split="train", base_dir=data_dir),
    }

    for task_name, ds in train_tasks.items():
        print(f"\nTask name: {task_name}")
        if task_name == "smoltalk":
            threshold_percentile = 99.9
        else:
            threshold_percentile = None
        encoded = tokenize_dataset(
            ds, tok_info, threshold_percentile=threshold_percentile
        )
        save_to_parquet(encoded, args.save_data_dir, task_name, split="train")

    val_tasks = {
        "smoltalk": SmolTalk(split="test"),  # 24K rows in test set
        "mmlu": MMLU(
            subset="all", split="test", stop=5200
        ),  # 14K rows in test set, use only 5.2K to match the train ratios
        "gsm8k": GSM8K(
            subset="main", split="test", stop=420
        ),  # 1.32K rows in test set, use only 420 to match the train ratios
    }  # total: 24K + 14K + 1.32K ~= 39K rows

    for task_name, ds in val_tasks.items():
        print(f"\nTask name: {task_name}")
        if task_name == "smoltalk":
            threshold_percentile = 99.9
        else:
            threshold_percentile = None
        encoded = tokenize_dataset(
            ds, tok_info, threshold_percentile=threshold_percentile
        )
        save_to_parquet(encoded, args.save_data_dir, task_name, split="test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for Mid-training")
    parser.add_argument(
        "--save_data_dir",
        help="Directory to save the tokenized records",
        default="/home/ubuntu/midtrain_data/",
    )
    parser.add_argument(
        "--records_per_shard",
        help="Number of records to write per shard",
        default=64_000,
        type=int,
    )
    parser.add_argument(
        "--mmlu_epochs", help="Number of passes for MMLU", default=1, type=int
    )
    parser.add_argument(
        "--gsm8k_epochs", help="Number of passes for GSM8K", default=1, type=int
    )
    args = parser.parse_args()
    main(args)
