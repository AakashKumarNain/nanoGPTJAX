############################################# Without grain ##################################################

import threading
import numpy as np
from pathlib import Path


def _load_data_shard(file: Path):
    header = np.fromfile(str(file), count=256, dtype=np.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = np.empty(
            num_tokens, dtype=np.uint16
        )  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens)  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


# This implementation here is based on modded-nanogpt dataloader implementation but wth some
# modifications. Here we do not use torch dataloader, and relies purely on a multi-threaded
# numpy dataloader. Though this dataloader is multi-threaded, there are still computational
# inefficiencies here where we see bubbles in the data pipeline, and the GPU sits idle.
# TODO: Replace it with an efficient grain dataloader

BOS_ID = 50256


class BOSFinder:
    def __init__(self, tokens):
        # Precompute BOS positions once per shard
        self.tokens = tokens
        self.size = len(tokens)
        self.bos_idx = np.where(tokens == BOS_ID)[0]
        self.i = 0
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = np.where(self.tokens == BOS_ID)[0]
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, batch_size: int, max_seq_len: int):
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


class DataPreloader:
    # Helper for asynchronously loading next shard and indexing bos tokens
    def __init__(self, file_iter, batch_size: int = 1):
        self.file_iter = file_iter
        self.batch_size = batch_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens))
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data


############################################# With grain ##################################################


# import grain
# import numpy as np
# from pathlib import Path
# from grain.multiprocessing import SharedMemoryArray


# BOS_ID = 50256

# class BOSFinder:
#     def __init__(self, tokens):
#         # Precompute BOS positions once per shard
#         self.tokens=tokens
#         self.size = len(tokens)
#         self.bos_idx = np.where(tokens == BOS_ID)[0]
#         self.i = 0
#         self.batch_iter = 0

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
#                     raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")

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


# class LoadShardTokens(grain.transforms.Map):
#     def map(self, path):
#         file = Path(path)

#         header = np.fromfile(str(file), count=256, dtype=np.int32)
#         assert header[0] == 20240520, "magic number mismatch in the data .bin file"
#         assert header[1] == 1, "unsupported version"
#         num_tokens = int(header[2])

#         with file.open("rb", buffering=0) as f:
#             f.seek(256 * 4)
#             tokens = SharedMemoryArray((num_tokens,), dtype=np.uint16)
#             nbytes = f.readinto(tokens)
#             assert nbytes == 2 * num_tokens, "number of tokens read does not match header"

#         bos_idx = np.flatnonzero(tokens == BOS_ID)
#         return {"path": str(file), "tokens": tokens, "bos_idx": bos_idx, "size": num_tokens}


# def make_grain_shard_loader(files, prefetch=2, worker_count=4):

#     # files should be a list of pathlib.Path or str
#     source = grain.sources.SharedMemoryDataSource([str(p) for p in files])
#     sampler = grain.samplers.SequentialSampler(num_records=len(source))
#     ops = [LoadShardTokens()]
#     return grain.DataLoader(
#         data_source=source,
#         sampler=sampler,
#         operations=ops,
#         worker_count=worker_count,
#         worker_buffer_size=prefetch,
#     )


# # def make_grain_shard_loader(files, prefetch=2, worker_count=1):
# #     from grain.experimental import ThreadPrefetchIterDataset
# #     ds = grain.MapDataset.source([str(p) for p in files]).map(LoadShardTokens()).to_iter_dataset()
# #     if prefetch > 0:
# #         ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=prefetch)
# #     return ds

# # def make_grain_shard_loader(files, prefetch=2, worker_count=1, num_threads=16, read_buffer_size=500):
# #     from grain.experimental import ThreadPrefetchIterDataset

# #     read_opts = grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=read_buffer_size)
# #     ds = grain.MapDataset.source([str(p) for p in files]).map(LoadShardTokens()).to_iter_dataset(read_options=read_opts)

# #     if prefetch > 0:
# #         ds = ThreadPrefetchIterDataset(ds, prefetch_buffer_size=prefetch)

# #     return ds
