# This file is generated using codex. This is a bit buggy/feature-incomplete
# and we aim to replace this with out own implementation n the future.

# We do a bit of prepreocessing here - find a the start and end of the sequences
# based on where the BOS token is. Once we have those boundaries, we save it
# on the disk so that we can some time during data loading in the training loop.
# Once we have downloaded the cached fineweb10B tokens, do the preprocessing as following:
# 
# files = sorted(glob.glob("fineweb10B/*train*.bin"))
# bos_id = 50256
# out_path = "fineweb_train_bos_index.npz"
# build_bos_doc_index(files, bos_id, out_path)

# # Build index for validation tokens
# files = sorted(glob.glob("fineweb10B/*val*.bin"))
# bos_id = 50256
# out_path = "fineweb_val_bos_index.npz"
# build_bos_doc_index(files, bos_id, out_path)
#

import os
import threading
import numpy as np
import grain
import numpy as np


HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4  # int32

MAGIC = 20240520
VERSION = 1


def read_header(path):
    with open(path, "rb") as f:
        hdr = np.fromfile(f, dtype=np.int32, count=HEADER_INTS)
    if hdr.size != HEADER_INTS:
        raise ValueError(f"{path}: too small for header")
    if int(hdr[0]) != MAGIC:
        raise ValueError(f"{path}: bad magic {int(hdr[0])}, expected {MAGIC}")
    if int(hdr[1]) != VERSION:
        raise ValueError(f"{path}: bad version {int(hdr[1])}, expected {VERSION}")
    n_tok = int(hdr[2])
    if n_tok < 0:
        raise ValueError(f"{path}: negative token count {n_tok}")
    return n_tok


def build_bos_doc_index(files, bos_id, out_path):
    bos_id = int(bos_id)

    tokens_per_shard = []
    total_tokens = 0

    shard_ids = []
    starts = []
    doc_ends = []

    for sid, path in enumerate(files):
        n_tok = read_header(path)
        tokens_per_shard.append(n_tok)
        total_tokens += n_tok

        mm = memmap_tokens(path, n_tok)
        bos_pos = np.flatnonzero(mm == bos_id).astype(np.int64)  # all doc starts

        if bos_pos.size == 0:
            continue

        # doc_end for each BOS is next BOS, last doc ends at shard end
        ends = np.empty_like(bos_pos)
        ends[:-1] = bos_pos[1:]
        ends[-1] = n_tok

        shard_ids.append(np.full(bos_pos.shape, sid, dtype=np.int32))
        starts.append(bos_pos)
        doc_ends.append(ends)

    shard_ids = np.concatenate(shard_ids) if shard_ids else np.zeros((0,), np.int32)
    starts    = np.concatenate(starts)    if starts    else np.zeros((0,), np.int64)
    doc_ends  = np.concatenate(doc_ends)  if doc_ends  else np.zeros((0,), np.int64)

    # os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        files=np.asarray(files),
        bos_id=np.int64(bos_id),
        total_tokens=np.int64(total_tokens),
        tokens_per_shard=np.asarray(tokens_per_shard, dtype=np.int64),
        shard_ids=shard_ids,
        starts=starts,
        doc_ends=doc_ends,
    )


def memmap_tokens(path, n_tok):
    return np.memmap("../"+ path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(n_tok,))



class BosDocIndexedSource(grain.sources.RandomAccessDataSource):
    def __init__(self, index_path, *, seqlen):
        z = np.load(index_path, allow_pickle=False)
        self.files = [str(x) for x in z["files"]]
        self.bos_id = int(z["bos_id"])
        self.total_tokens = int(z["total_tokens"])
        self.tokens_per_shard = z["tokens_per_shard"].astype(np.int64)

        self.shard_ids = z["shard_ids"].astype(np.int32)
        self.starts = z["starts"].astype(np.int64)
        self.doc_ends = z["doc_ends"].astype(np.int64)

        self.seqlen = int(seqlen)
        self.need = self.seqlen + 1

        mask = (self.starts + self.need) <= self.doc_ends
        self.valid = np.flatnonzero(mask).astype(np.int64)

        # Thread-local cache: each thread gets its own (sid, mm)
        self._tls = threading.local()

        self.num_examples = int(self.valid.size)
        self.total_tokens = self.num_examples * self.seqlen
        self.total_raw_tokens = self.num_examples * (self.seqlen + 1)

    def __len__(self):
        return int(self.valid.size)

    def _mm(self, sid):
        sid = int(sid)

        cached_sid = getattr(self._tls, "sid", None)
        cached_mm = getattr(self._tls, "mm", None)
        if cached_sid == sid and cached_mm is not None:
            return cached_mm

        path = self.files[sid]
        n_tok = int(self.tokens_per_shard[sid])
        mm = memmap_tokens(path, n_tok)

        self._tls.sid = sid
        self._tls.mm = mm
        return mm

    def __getitem__(self, k):
        i = int(self.valid[k])
        sid = int(self.shard_ids[i])
        s = int(self.starts[i])
        e = s + self.need

        tok = np.asarray(self._mm(sid)[s:e], dtype=np.int64)
        x = tok[:-1]
        y = tok[1:]
        if x[0] != self.bos_id:
            raise ValueError("BOS misalignment: index/bos_id mismatch")
        return x, y


def make_grain_iter(index_path, seqlen, batch_size, shuffle=True, seed=0,
                    num_threads=16, prefetch_buffer_size=512, drop_remainder=True):
    
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index: {index_path} was not found. Did you build the index before running this?")
    
    source = BosDocIndexedSource(index_path, seqlen=seqlen)
    ds = grain.MapDataset.source(source)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder).repeat()
    ds = ds.to_iter_dataset(grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=prefetch_buffer_size))
    ds = iter(ds)
    return source, ds