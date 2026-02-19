"""This dataloader is specifically build for mid-training/SFT stages. Here are a
few noticeable things about this:

- 
"""

import os
import json
import grain
import tiktoken
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from array_record.python import array_record_module



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
        user_start, user_end,
        assistant_start, assistant_end,
        system_start, system_end,
        tool_start, tool_end,
        pad_token,
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



ROLE_MAP = {
    "system":    ("system_start",    "system_end"),
    "user":      ("user_start",      "user_end"),
    "assistant": ("assistant_start", "assistant_end"),
    "tool":      ("tool_start",      "tool_end"),
}


def format_conversation(example, tok_info):
    """Format a single or multi-turn chat example into a string ready for tokenization."""
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

    roles = {m.get("role") for m in messages}

    if "user" not in roles or "assistant" not in roles:
        return None

    return {"text": "".join(parts)}


def tokenize(example, tok_info):
    """Tokenize text and bake the completion mask into the sign bit."""
    tokenizer = tok_info["tokenizer"]
    tokens = np.array(
        tokenizer.encode(example["text"],allowed_special="all"),
        dtype=np.int32,
    )

    ast_start = tok_info["assistant_start_id"]
    ast_end = tok_info["assistant_end_id"]

    mask = np.zeros(len(tokens), dtype=np.bool_)

    inside_assistant = False
    for i, tok in enumerate(tokens):
        if tok == ast_start:
            inside_assistant = True
            continue
        if tok == ast_end:
            mask[i] = True
            inside_assistant = False
            continue
        if inside_assistant:
            mask[i] = True

    return {"input_ids": encode_mask_into_ids(tokens, mask)}


# We will save the parquet files in this format
PARQUET_SCHEMA = pa.schema([("input_ids", pa.list_(pa.int32())),])


def save_tokenized(
        dataset_path,
        data_dir,
        tok_info,
        split,
        records_per_shard, subset="openhermes-100k"
    ):
    """Stream-download, tokenize, sign-encode, and write Parquet shards."""

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    name = dataset_path.split("/")[-1]
    prefix = f"{name}_{subset}" if subset and subset != "all" else name

    if subset and subset != "all":
        ds = load_dataset(dataset_path, subset, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_path, split=split, streaming=True)

    shard_idx = 0
    total_written = 0
    skipped = 0
    buffer = []

    def shard_path(idx):
        return os.path.join(data_dir, f"{prefix}_{split}-{idx:05d}.parquet")

    def flush_buffer():
        nonlocal shard_idx
        if not buffer:
            return
        table = pa.table({"input_ids": buffer}, schema=PARQUET_SCHEMA)
        pq.write_table(table, shard_path(shard_idx))
        buffer.clear()
        shard_idx += 1

    pbar = tqdm(desc=f"Tokenizing {prefix}/{split}", unit=" examples")

    for example in ds:
        formatted = format_conversation(example, tok_info)
        if formatted is None:
            skipped += 1
            pbar.set_postfix(written=total_written, skipped=skipped, shard=shard_idx)
            pbar.update(1)
            continue

        tokenized = tokenize(formatted, tok_info)
        buffer.append(tokenized["input_ids"].tolist())
        total_written += 1

        pbar.set_postfix(written=total_written, skipped=skipped, shard=shard_idx)
        pbar.update(1)

        if len(buffer) >= records_per_shard:
            flush_buffer()

    # Flush remaining — no minimum size constraint with Parquet
    flush_buffer()
    pbar.close()

    meta = {
        "dataset": dataset_path,
        "subset": subset,
        "split": split,
        "dtype": "int32",
        "total_records": total_written,
        "num_shards": shard_idx,
        "bos_token_id": tok_info["bos_id"],
    }
    meta_path = os.path.join(data_dir, f"{prefix}_{split}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote {total_written} records across {shard_idx} shard(s) to {data_dir}")
    print(f"Skipped {skipped} examples")
    print(f"Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# Grain loader helpers
# ---------------------------------------------------------------------------

def get_shard_paths(data_dir, split="train"):
    """Return sorted list of Parquet shard paths for the given split."""
    paths = sorted(Path(data_dir).glob(f"*{split}*.parquet"))
    print(f"Number of files found: ", len(paths))
    return [str(p) for p in paths]

def load_meta(data_dir, split="train"):
    for fname in os.listdir(data_dir):
        if fname.endswith(f"{split}_meta.json"):
            with open(os.path.join(data_dir, fname)) as f:
                return json.load(f)
    raise FileNotFoundError(f"No {split}_meta.json found in {data_dir}")


def load_all_token_rows(shard_paths):
    """Read all Parquet shards into a list of np.int32 arrays."""
    rows = []
    for path in shard_paths:
        table = pq.read_table(path, columns=["input_ids"])
        for row in table["input_ids"]:
            rows.append(np.array(row.as_py(), dtype=np.int32))
    return rows


class ParquetTokenSource:
    """A Grain-compatible random-access source backed by Parquet shards.

    Reads all shards into memory once (token IDs are small),
    then serves individual rows by index.
    """

    def __init__(self, shard_paths):
        self._rows = load_all_token_rows(shard_paths)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return {"input_ids": self._rows[idx]}


def make_grain_shard_loader(
    batch_size,
    sequence_length,
    grad_accum_steps,
    data_sharding,
    data_dir,
    split="train",
    repeat=True,
    cpu_buffer_size=16,
    device_buffer_size=4,
):
    shard_paths = get_shard_paths(data_dir, split)
    meta = load_meta(data_dir, split)
    source = ParquetTokenSource(shard_paths)

    ds = grain.MapDataset.source(source)
    if repeat:
        ds = ds.shuffle(1234).repeat()

    total_batch_size = grad_accum_steps * batch_size if grad_accum_steps > 1 else batch_size

    ds = grain.experimental.ConcatThenSplitIterDataset(
        parent=ds,
        length_struct={"input_ids": sequence_length},
        split_full_length_features=True,
        bos_handling=grain.experimental.BOSHandling.REPLACE_FIRST_TOKEN_WITH_BOS,
        bos_features=("input_ids",),
        bos_token_id=meta["bos_token_id"],
    ).batch(total_batch_size, drop_remainder=True)

    # Recover completion_mask from sign bit
    ds = ds.map(decode_mask_from_ids)

    if grad_accum_steps > 1:
        ds = ds.map(partial(prepare_train_accum_batch, grad_accum_steps=grad_accum_steps))
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
    print(f"Vocab size: {tok_info['vocab_size']}")
    print(f"BOS id:     {tok_info['bos_id']}")
    print(f"Assistant start id: {tok_info['assistant_start_id']}")

    save_tokenized(
        args.dataset_path,
        args.save_data_dir, 
        tok_info,
        split=args.split,
        records_per_shard=args.records_per_shard
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for Mid-training")
    parser.add_argument("--dataset_path", help="Name of the dataset (on HF)", required=True)
    parser.add_argument("--save_data_dir", help="Directory to save the tokenized records", required=True, default="/home/ubuntu/nanochat/jaxnano/sft_data/")
    parser.add_argument("--records_per_shard", help="Number of records to write per shard", default=10_0000, type=int)
    parser.add_argument("--split", help="Data split to download", required=True)
    args = parser.parse_args()
    main(args)