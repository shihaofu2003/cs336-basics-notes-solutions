import mmap
import os
import pathlib
import time
from multiprocessing import Pool
from typing import Iterable, Iterator, List
import json

import numpy as np
import regex as re
from tqdm import tqdm

from cs336_basics.tokenization.prepare import pre_tokenize, load_vocab_merges, find_chunk_boundaries


class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_bytes_id = None
        self.merges = merges
        self.special_tokens = special_tokens

        if self.vocab_bytes_id is None:
            self._load_vocab_json()

        self.bpe_ranks = {}
        for rank, (a, b) in enumerate(merges):
            self.bpe_ranks[(a, b)] = rank

    def _load_vocab_json(self):
        self.vocab_bytes_id = {
            vocab_item: vocab_index
            for vocab_index, vocab_item in self.vocab.items()
        }

    def _bpe_merge(self, pieces: list[bytes]) -> list[bytes]:
        while len(pieces) > 1:
            # 找所有相邻 pair，找出 rank 最小（优先级最高）的那个
            pairs = [(self.bpe_ranks.get((pieces[i], pieces[i + 1]), float('inf')), i)
                     for i in range(len(pieces) - 1)]
            if not pairs:
                break
            min_rank, min_idx = min(pairs)  # ← 关键！只找最优的
            if min_rank == float('inf'):
                break

            # 执行一次合并
            pieces[min_idx] = pieces[min_idx] + pieces[min_idx + 1]
            del pieces[min_idx + 1]

        return pieces

    @classmethod
    def from_files(cls,
                   vocab_filepath: str,
                   merges_filepath: str,
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_json = json.load(f)
            vocab = {
                vocab_index: vocab_item.encode("utf-8")
                for vocab_item, vocab_index in vocab_json.items()
            }

        with open(merges_filepath, encoding="utf-8") as f:
            tokens_merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    merge_token_1.encode(encoding="utf-8"),
                    merge_token_2.encode(encoding="utf-8"),
                )
                for merge_token_1, merge_token_2 in tokens_merges
            ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        text_indices: list[int] = []
        if text == "":
            return text_indices

        # split_with_special_tokens
        if not self.special_tokens:
            parts = [text]
        else:
            escaped = sorted((re.escape(tok) for tok in self.special_tokens), key=len, reverse=True)
            special_pat = "(" + "|".join(escaped) + ")"
            parts = re.split(special_pat, text)

        # loop for parts
        # print("--parts: ", parts)
        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                text_indices.append(self.vocab_bytes_id[part.encode()])
            else:
                # pre_tokenization
                bytes_words: list[bytes] = pre_tokenize(part)

                for i in range(len(bytes_words)):
                    bytes_word: list[bytes] = [bytes([b]) for b in bytes_words[i]]
                    split_bytes_word: list[bytes] = self._bpe_merge(bytes_word)

                    for split_bytes in split_bytes_word:
                        text_indices.append(self.vocab_bytes_id[split_bytes])

        return text_indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for item in iterable:
            # words = split_words_with_special(item, self.special_tokens)
            # for word in words:
            yield from self.encode(item)

    def encode_iterable_extend(self, iterable: Iterable[str]):
        for item in iterable:
            # print(item)
            # words = split_words_with_special(item, self.special_tokens)
            # for word in words:
            yield self.encode(item)

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""

        bytes_text: bytes = bytearray()
        for item in ids:
            bytes_text += self.vocab[item]

        return bytes_text.decode('utf-8', errors="replace")


def encode_file(file_path, vocab_path, merges_path, npy_path):
    vocab, merges = load_vocab_merges(vocab_path, merges_path)
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    st = time.perf_counter()
    with open(file_path, "r", encoding="utf-8") as f:
        # 先获取总行数（可选，用于更精确的进度条）
        total_lines = sum(1 for _ in open(file_path, encoding="utf-8"))
        f.seek(0)  # 重置文件指针

        ids = []
        # tqdm 包裹 tokenizer.encode_iterable(f)
        for _id in tqdm(
                tokenizer.encode_iterable_extend(f),
                total=total_lines,
                desc="Tokenizing",
                unit="line",
                colour="green",
                leave=True
        ):
            ids.extend(_id)

    end = time.perf_counter()
    print(f"Encoded {len(ids):,} tokens in {end - st:.3f}s")
    print(f"Speed: {len(ids) / (end - st):.0f} tokens/sec")

    npy_path = npy_path.parent / f"{npy_path.stem}_{len(ids)}.npy"
    np.save(npy_path, np.array(ids, dtype=np.uint16))
    print(f"Saved to file: {npy_path}")


_global_tokenizer: Tokenizer | None = None
_mm = None


def _worker_init(vocab_path, merges_path, file_path):
    """每个子进程启动时自动执行一次"""
    global _global_tokenizer
    global _mm
    vocab, merges = load_vocab_merges(vocab_path, merges_path)
    _global_tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])

    with open(file_path, "rb") as f:
        _mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)


def _encode_chunk(args):
    """子进程实际执行的函数"""
    chunk_id, chunk_start, chunk_end, npy_path = args
    global _global_tokenizer
    global _mm

    chunk = _mm[chunk_start: chunk_end].decode("utf-8", errors="ignore")
    ids_tokens = _global_tokenizer.encode(chunk)
    token_data = np.array(ids_tokens, dtype=np.uint16)

    npy_chunk_path = npy_path.parent / f"chunks" / f"{npy_path.stem}_chunk_{chunk_id:05d}.npy"
    np.save(npy_chunk_path, token_data)

    metadata = {"npy_chunk_path": str(npy_chunk_path.stem),
                "chunk_id": chunk_id,
                "token_count": len(token_data),
                "byte_size_GB": os.path.getsize(npy_chunk_path) / (2 ** 30)}

    return metadata


def encode_file_parallelism(file_path, vocab_path, merges_path, npy_path, num_chunks, num_processes):
    st = time.perf_counter()

    # 计算 chunks 的边界
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=num_chunks, split_special_token=b'<|endoftext|>')
    f.close()

    metadata = {"total_tokens": 0, "chunks": []}
    num_chunks = len(boundaries) - 1
    with Pool(
            processes=num_processes,
            initializer=_worker_init,
            initargs=(vocab_path, merges_path, file_path),
            maxtasksperchild=1
    ) as pool:
        args = [(i, boundaries[i], boundaries[i + 1], npy_path) for i in range(num_chunks)]
        for metadata_chunk in tqdm(pool.imap_unordered(_encode_chunk, args),
                                   total=num_chunks, desc="Tokenize chunks"):
            metadata["chunks"].append(metadata_chunk)
            metadata["total_tokens"] += metadata_chunk["token_count"]

    end = time.perf_counter()
    print(f"Encoded {metadata['total_tokens']} tokens in {end - st:.3f}s")
    print(f"Speed: {metadata['total_tokens'] / (end - st):.0f} tokens/sec")

    # validate metadata
    ids = []
    for chunk in metadata["chunks"]:
        ids.append(chunk["chunk_id"])
    assert len(set(ids)) == num_chunks

    metadata_path = npy_path.parent / f"{npy_path.stem}_chunk_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata Saved to file: {metadata_path}")

    tokens_cnt = 0
    limited_cnt = 2 ** 30
    shard_tokens = None
    shard_id = 0
    metadata_shards = {"total_tokens": metadata['total_tokens'], "limited_tokens": limited_cnt, "shards": []}
    metadata_chunks = sorted(metadata["chunks"], key=lambda x: x["chunk_id"])
    for metadata_chunk in tqdm(metadata_chunks, total=len(metadata_chunks), desc="Chunk Merging"):
        chunk_id = metadata_chunk["chunk_id"]
        tokens_cnt += metadata_chunk["token_count"]
        npy_chunk_path = npy_path.parent / f"chunks" / f"{metadata_chunk['npy_chunk_path']}.npy"

        if chunk_id == 0:
            shard_tokens = np.load(npy_chunk_path)
        else:
            shard_tokens = np.hstack((shard_tokens, np.load(npy_chunk_path)))

        if tokens_cnt >= limited_cnt or chunk_id == num_chunks - 1:
            npy_shard_path = npy_path.parent / f"shards" / f"{npy_path.stem}_shard_{shard_id:05d}.npy"
            np.save(npy_shard_path, shard_tokens[:limited_cnt])
            metadata_shards["shards"].append({
                "npy_chunk_path": str(npy_shard_path.stem),
                "token_count": tokens_cnt,
                "byte_size_GB": os.path.getsize(npy_shard_path) / (2 ** 30)
            })

            shard_tokens = shard_tokens[limited_cnt:]
            tokens_cnt = 0
            shard_id += 1

    metadata_shards_path = npy_path.parent / f"{npy_path.stem}_shard_metadata.json"
    with open(metadata_shards_path, "w", encoding="utf-8") as f:
        json.dump(metadata_shards, f, indent=4)
    print(f"Metadata Saved to file: {metadata_shards_path}")



if __name__ == "__main__":
    FIXTURES_PATH = pathlib.Path(__file__).resolve().parent.parent.parent

    # file_path = FIXTURES_PATH / "data/TinyStories/TinyStoriesV2-GPT4-train.txt"
    # vocab_path = FIXTURES_PATH / "data/TinyStories/TinyStories_train_vocab.json"
    # merges_path = FIXTURES_PATH / "data/TinyStories/TinyStories_train_vocab_merges.txt"
    # npy_path = FIXTURES_PATH / "data/TinyStories/TinyStories_train.npy"
    # # encode_file(file_path, vocab_path, merges_path, npy_path)
    # encode_file_parallelism(file_path, vocab_path, merges_path, npy_path, 1000, 10)

    file_path = FIXTURES_PATH / "data/owt/owt_train.txt"
    vocab_path = FIXTURES_PATH / "data/owt/owt_train_vocab.json"
    merges_path = FIXTURES_PATH / "data/owt/owt_train_vocab_merges.txt"
    npy_path = FIXTURES_PATH / "data/owt/owt_train.npy"
    # encode_file(file_path, vocab_path, merges_path, npy_path)
    encode_file_parallelism(file_path, vocab_path, merges_path, npy_path, 4000, 300)