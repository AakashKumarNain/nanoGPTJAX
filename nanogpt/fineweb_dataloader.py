############################################# Without grain ##################################################

# import os
# import threading
# import numpy as np
# import grain
# import numpy as np
# import argparse
# from pathlib import Path


# def _load_data_shard(file: Path):
#     header = np.fromfile(str(file), count=256, dtype=np.int32) # header is 256 int32
#     assert header[0] == 20240520, "magic number mismatch in the data .bin file"
#     assert header[1] == 1, "unsupported version"
#     num_tokens = int(header[2]) # number of tokens (claimed)
#     with file.open("rb", buffering=0) as f:
#         tokens = np.empty(num_tokens, dtype=np.uint16) # avoid pin_memory copy by @YouJiacheng
#         f.seek(256 * 4)
#         nbytes = f.readinto(tokens) # avoid bytes->array copy by @YouJiacheng
#         assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
#     return tokens

# BOS_ID = 50256


# class BOSFinder:
#     def __init__(self, tokens):
#         # Precompute BOS positions once per shard
#         self.tokens=tokens
#         self.size = len(tokens)
#         self.bos_idx = np.where(tokens == BOS_ID)[0]
#         self.i = 0
#         self.batch_iter = 0

#     def _load(self):
#         self.bos_idx_async = np.where(self.tokens == BOS_ID)[0]
#         self.ready.set()

#     def start(self):
#         self.ready.clear()
#         self.thread = threading.Thread(target=self._load)
#         self.thread.start()

#     def get(self):
#         if self.thread:
#             self.ready.wait()
#             self.thread.join()
#         self.bos_idx = self.bos_idx_async

#     def next_batch(self, batch_size: int, max_seq_len: int):
#         n = len(self.bos_idx)
#         starts = []
#         ends = []

#         idx = self.i
#         for i in range(batch_size):
#             cur_len = 0
#             target_len = max_seq_len + 1

#             while cur_len < target_len:
#                 if idx >= n:
#                     raise StopIteration("Insufficient BOS ahead; hit tail of shard.")

#                 cur = self.bos_idx[idx]
#                 starts.append(cur)

#                 remaining = target_len - cur_len
#                 next_bos = self.bos_idx[idx + 1] if idx + 1 < n else self.size

#                 # Take either remaining tokens or up to next BOS
#                 end = min(next_bos, cur + remaining)
#                 ends.append(end)

#                 cur_len += end - cur
#                 idx += 1

#             assert cur_len == target_len

#         self.i = idx
#         self.batch_iter += 1
#         return starts, ends

# class DataPreloader:
#     # Helper for asynchronously loading next shard and indexing bos tokens
#     def __init__(self, file_iter, batch_size: int = 1):
#         self.file_iter = file_iter
#         self.batch_size = batch_size
#         self.thread = None
#         self.data = None
#         self.ready = threading.Event()

#     def _load(self):
#         tokens = _load_data_shard(next(self.file_iter))
#         self.data = (tokens, BOSFinder(tokens))
#         self.ready.set()

#     def start(self):
#         self.ready.clear()
#         self.thread = threading.Thread(target=self._load)
#         self.thread.start()

#     def get(self):
#         if self.thread:
#             self.ready.wait()
#             self.thread.join()
#         return self.data

############################################# With grain ##################################################


import grain
import numpy as np
from pathlib import Path
from grain.multiprocessing import SharedMemoryArray


BOS_ID = 50256


class BOSFinder:
    def __init__(self, tokens):
        # Precompute BOS positions once per shard
        self.tokens = tokens
        self.size = len(tokens)
        self.bos_idx = np.where(tokens == BOS_ID)[0]
        self.i = 0
        self.batch_iter = 0
        # lazy: do not allocate index until build(...) is called
        self.built_ready = False

    def build(self, batch_size, max_seq_len):
        # Precompute a static index of all (start, end) pairs for every full batch in this shard
        # starting from the beginning of the shard. Resets iteration counters.
        n = len(self.bos_idx)
        target_len = max_seq_len + 1
        starts = []
        ends = []
        ptrs = [0]  # cumulative count of pairs; one entry per completed batch
        idx = 0  # cursor into bos_idx

        while True:
            batch_pairs_begin = len(starts)
            full_batch = True
            for i in range(batch_size):
                cur_len = 0
                while cur_len < target_len:
                    if idx >= n:
                        full_batch = False
                        break
                    cur = self.bos_idx[idx]
                    starts.append(cur)

                    remaining = target_len - cur_len
                    next_bos = self.bos_idx[idx + 1] if idx + 1 < n else self.size

                    end = min(next_bos, cur + remaining)
                    ends.append(end)

                    cur_len += end - cur
                    idx += 1
                if not full_batch:
                    break
                assert cur_len == target_len
            if not full_batch:
                # rollback any partial pairs for the incomplete batch
                del starts[batch_pairs_begin:]
                del ends[batch_pairs_begin:]
                break
            ptrs.append(len(starts))

        # store as numpy for compactness and fast slicing
        self.built_starts = np.asarray(starts, dtype=np.int32)
        self.built_ends = np.asarray(ends, dtype=np.int32)
        self.built_ptrs = np.asarray(ptrs, dtype=np.int64)
        self.built_batch_size = batch_size
        self.built_max_seq_len = max_seq_len
        self.built_ready = True
        self.i = 0
        self.batch_iter = 0
        return len(self.built_ptrs) - 1  # number of full batches indexed

    def next_batch(self, batch_size: int, max_seq_len: int):
        # Fast path: use prebuilt index if available and matching request
        if (
            self.built_ready
            and self.built_batch_size == batch_size
            and self.built_max_seq_len == max_seq_len
        ):
            b = self.batch_iter
            if b >= len(self.built_ptrs) - 1:
                raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
            p0 = int(self.built_ptrs[b])
            p1 = int(self.built_ptrs[b + 1])
            starts = self.built_starts[p0:p1].tolist()
            ends = self.built_ends[p0:p1].tolist()
            # keep original counters consistent
            self.i += p1 - p0
            self.batch_iter += 1
            return starts, ends

        # Original on-the-fly path (unchanged logic)
        n = len(self.bos_idx)
        starts = []
        ends = []

        idx = self.i
        for i in range(batch_size):
            cur_len = 0
            target_len = max_seq_len + 1

            while cur_len < target_len:
                if idx >= n:
                    raise StopIteration("Insufficient BOS ahead; hit tail of shard.")

                cur = self.bos_idx[idx]
                starts.append(cur)

                remaining = target_len - cur_len
                next_bos = self.bos_idx[idx + 1] if idx + 1 < n else self.size

                # Take either remaining tokens or up to next BOS
                end = min(next_bos, cur + remaining)
                ends.append(end)

                cur_len += end - cur
                idx += 1

            assert cur_len == target_len

        self.i = idx
        self.batch_iter += 1
        return starts, ends


class CustomSharedMemoryDataSource(grain.sources.SharedMemoryDataSource):
    def __init__(self, elements=None, *, name=None):
        if elements is not None:
            elements = [str(Path(p).resolve()) for p in elements]
        super().__init__(elements, name=name)
        self.files = [] if elements is None else elements
        self.name = name

    def __repr__(self):
        return f"Fineweb10BSharedMemoryData(name={self.name}, len={len(self.files)})"


class LoadShardTokens(grain.transforms.Map):
    def map(self, path):
        file = Path(path)

        header = np.fromfile(str(file), count=256, dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        num_tokens = int(header[2])

        with file.open("rb", buffering=0) as f:
            f.seek(256 * 4)
            tokens = SharedMemoryArray((num_tokens,), dtype=np.uint16)
            nbytes = f.readinto(tokens)
            assert nbytes == 2 * num_tokens, (
                "number of tokens read does not match header"
            )

        bos_idx = np.flatnonzero(tokens == BOS_ID)
        return {
            "path": str(file),
            "tokens": tokens,
            "bos_idx": bos_idx,
            "size": num_tokens,
        }


def make_grain_shard_loader(files, worker_count=4, read_options=None, custom=False):
    # files should be a list of pathlib.Path or str
    # source = grain.sources.SharedMemoryDataSource([str(p) for p in files])
    if custom:
        files = [str(p) for p in files]
        source = CustomSharedMemoryDataSource(files, name="fineweb10b")
    else:
        files = [str(p) for p in files]
        source = grain.sources.SharedMemoryDataSource(files)
    sampler = grain.samplers.SequentialSampler(num_records=len(source))
    ops = [LoadShardTokens()]
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=worker_count,
        read_options=read_options,
    )


# def make_grain_shard_loader(files, prefetch=2, worker_count=1):
#     from grain.experimental import ThreadPrefetchIterDataset
#     ds = grain.MapDataset.source([str(p) for p in files]).map(LoadShardTokens()).to_iter_dataset()
#     if prefetch > 0:
#         ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=prefetch)
#     return ds

# def make_grain_shard_loader(files, prefetch=2, worker_count=1, num_threads=16, read_buffer_size=500):
#     from grain.experimental import ThreadPrefetchIterDataset

#     read_opts = grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=read_buffer_size)
#     ds = grain.MapDataset.source([str(p) for p in files]).map(LoadShardTokens()).to_iter_dataset(read_options=read_opts)

#     if prefetch > 0:
#         ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=prefetch)

#     return ds
