# This file is generated using codex. This is a bit buggy/feature-incomplete
# and we aim to replace this with out own implementation n the future.

import os
import glob
import random
import numpy as np
import multiprocessing as mp


LEGACY_MAGIC = 20240520
MAGIC = 278895051
HEADER_SIZE = 256
HEADER_BYTES = HEADER_SIZE * 4


class BinaryShard:
    def __init__(self, path, use_bos_index=True):
        self.path = path
        header = np.fromfile(path, dtype=np.int32, count=HEADER_SIZE)
        if header.size != HEADER_SIZE:
            raise ValueError(f"bad header in {path}")
        magic = int(header[0])
        num_tokens = int(header[2])
        if magic == LEGACY_MAGIC:
            itemsize = 2
        elif magic == MAGIC:
            itemsize = int(header[3])
            if itemsize not in (2, 4):
                raise ValueError(f"bad dtype size {itemsize} in {path}")
        else:
            raise ValueError(f"bad magic {magic} in {path}")
        dtype = np.uint16 if itemsize == 2 else np.uint32
        self.tokens = np.memmap(
            path, mode="r", dtype=dtype, offset=HEADER_BYTES, shape=(num_tokens,)
        )
        self.n = num_tokens
        self.itemsize = itemsize
        self.bos_positions = None
        if use_bos_index:
            bos_idx_path = path + ".bos.idx"
            if os.path.exists(bos_idx_path):
                bos_bytes = np.memmap(bos_idx_path, mode="r", dtype=np.int32)
                tok_pos = (
                    np.asarray(bos_bytes, dtype=np.int64) - HEADER_BYTES
                ) // self.itemsize
                tok_pos = tok_pos[(tok_pos >= 0) & (tok_pos < self.n)]
                self.bos_positions = tok_pos


def sample_unaligned(rng, shard, seq_len):
    n = shard.n
    start = rng.randrange(0, n - seq_len - 1)
    x = np.asarray(shard.tokens[start : start + seq_len])
    y = np.asarray(shard.tokens[start + 1 : start + 1 + seq_len])
    return x, y


def sample_aligned_with_index(rng, shard, seq_len):
    valid = shard.bos_positions
    if valid is None or valid.size == 0:
        return None
    max_start = shard.n - seq_len - 1
    j = rng.randrange(valid.size)
    for _ in range(min(8, valid.size)):
        pos = int(valid[j])
        if pos <= max_start:
            x = np.asarray(shard.tokens[pos : pos + seq_len])
            y = np.asarray(shard.tokens[pos + 1 : pos + 1 + seq_len])
            return x, y
        j = (j + 1) % valid.size
    return None


def sample_aligned_linear(rng, shard, seq_len, bos_token):
    n = shard.n
    start_guess = rng.randrange(0, n - seq_len - 1)
    search_slice = shard.tokens[start_guess : n - seq_len]
    hits = np.where(search_slice == bos_token)[0]
    if hits.size == 0:
        return None
    start = int(start_guess + hits[0])
    x = np.asarray(shard.tokens[start : start + seq_len])
    y = np.asarray(shard.tokens[start + 1 : start + 1 + seq_len])
    return x, y


def worker_loop(cfg, shard_paths, data_q, stop_event):
    rng = random.Random(cfg["seed"])
    shards = []
    for p in shard_paths:
        try:
            s = BinaryShard(p, use_bos_index=cfg["use_bos_index"])
            if s.n > cfg["seq_len"] + 1:
                shards.append(s)
        except Exception:
            continue
    if not shards:
        raise RuntimeError("worker has no usable shards")

    while not stop_event.is_set():
        xs = []
        ys = []
        attempts = 0
        while len(xs) < cfg["batch_size"] and not stop_event.is_set():
            attempts += 1
            shard = shards[rng.randrange(len(shards))]
            if not cfg["align_to_bos"]:
                x, y = sample_unaligned(rng, shard, cfg["seq_len"])
                xs.append(x)
                ys.append(y)
                continue
            out = None
            if shard.bos_positions is not None:
                out = sample_aligned_with_index(rng, shard, cfg["seq_len"])
            if out is None:
                out = sample_aligned_linear(
                    rng, shard, cfg["seq_len"], cfg["bos_token"]
                )
            if out is None:
                if attempts > cfg["batch_size"] * 50:
                    attempts = 0
                continue
            x, y = out
            xs.append(x)
            ys.append(y)

        if stop_event.is_set():
            break
        try:
            x_batch = np.stack(xs, axis=0)
            y_batch = np.stack(ys, axis=0)
            data_q.put((x_batch, y_batch), timeout=1.0)
        except Exception:
            if stop_event.is_set():
                break


class BinaryTokenDataLoader:
    def __init__(
        self,
        file_patterns,
        seq_len,
        batch_size,
        bos_token=50256,
        align_to_bos=True,
        seed=0,
        use_bos_index=True,
        num_workers=4,
        prefetch_batches=16,
    ):
        if isinstance(file_patterns, str):
            files = sorted(glob.glob(file_patterns))
        else:
            files = []
            for p in file_patterns:
                files.extend(glob.glob(p))
            files = sorted(files)
        if not files:
            raise FileNotFoundError("no .bin files matched")

        if os.name == "posix":
            ctx = mp.get_context("fork")
        else:
            ctx = mp.get_context("spawn")

        self.cfg = dict(
            seq_len=int(seq_len),
            batch_size=int(batch_size),
            bos_token=int(bos_token),
            align_to_bos=bool(align_to_bos),
            use_bos_index=bool(use_bos_index),
            seed=int(seed),
        )
        self.num_workers = int(num_workers)
        self.data_q = ctx.Queue(maxsize=int(prefetch_batches))
        self.stop_event = ctx.Event()

        shards_per_worker = [[] for _ in range(self.num_workers)]
        for i, f in enumerate(files):
            shards_per_worker[i % self.num_workers].append(f)
        self.procs = []
        for i in range(self.num_workers):
            cfg_i = dict(self.cfg)
            cfg_i["seed"] = self.cfg["seed"] + i + 1
            p = ctx.Process(
                target=worker_loop,
                args=(cfg_i, shards_per_worker[i], self.data_q, self.stop_event),
                daemon=True,
            )
            p.start()
            self.procs.append(p)

    def get_batch(self):
        return self.data_q.get()

    def close(self):
        self.stop_event.set()
        try:
            while not self.data_q.empty():
                try:
                    self.data_q.get_nowait()
                except Exception:
                    break
        except Exception:
            pass
        for p in self.procs:
            if p.is_alive():
                p.join(timeout=1.0)
        for p in self.procs:
            if p.is_alive():
                p.terminate()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
