import os
from typing import BinaryIO
import regex as re
from collections import defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def split_special_tokens(
        chunk: str,
        special_tokens: list[str]
) -> str:
    """
    Split special toknes from a chunk of text.
    """

    for special_token in special_tokens:
        chunk = chunk.replace(special_token, "")
    return chunk

def pre_tokenize(
        chunk: str
) -> list[bytes]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    iters = re.findall(PAT, chunk)
    bytes_words = [iter.encode() for iter in iters]
    return bytes_words

## Usage

special_tokens = ["<|endoftext|>"]
vocab_size = 256
vocab = {i: chr(i).encode() for i in range(256)}
for special_token in special_tokens:
    vocab[vocab_size] = special_token.encode()
    vocab_size += 1

merges: list[tuple[bytes, bytes]] = []
indices_merges: list[tuple[int, int]] = []

with open("../../data/TinyStories/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    num_processes = 3
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    print(boundaries)

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        print(len(chunk), chunk[:20])

        # Run pre-tokenization on your chunk and store the counts for each pre-token
        # remove special tokens
        chunk = split_special_tokens(chunk, special_tokens)
        bytes_words = pre_tokenize(chunk)
        print(bytes_words[:10], len(bytes_words))

        bytes_word_counts = defaultdict(int)  # {b"abc": 1}
        for bytes_word in bytes_words:
            bytes_word_counts[bytes_word] += 1

        indices_bytes_words = {bytes_word: list(map(int, bytes_word)) for bytes_word in bytes_word_counts.keys()}
        print(indices_bytes_words[bytes_words[1]])

        pair_counts = defaultdict(int)
        for bytes_word in bytes_word_counts.keys():
            indices_bytes_word = indices_bytes_words[bytes_word]

            for pair in zip(indices_bytes_word, indices_bytes_word[1:]):
                pair_counts[pair] += bytes_word_counts[bytes_word]

        pair_max = max(pair_counts)
        print(pair_max, pair_counts[pair_max], len(pair_counts))

        # merge
        index1, index2 = pair_max
        new_index = vocab_size

        vocab[vocab_size] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))
        indices_merges.append((index1, index2))
        vocab_size += 1

        for bytes_word, indices in indices_bytes_words.items():
            new_indices = []
            j = 0
            while j < len(indices):
                if j + 1 < len(indices) and indices[j] == index1 and indices[j+1] == index2:
                    new_indices.append(new_index)
                    j += 2
                else:
                    new_indices.append(indices[j])
                    j += 1
            indices_bytes_words[bytes_word] = new_indices

        print(merges)
        print(indices_merges)



        print("\n\n")

        break







