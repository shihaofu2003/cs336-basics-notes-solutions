import os
from typing import BinaryIO
import regex as re
import json
import pickle


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

    iters = re.finditer(PAT, chunk)
    bytes_words = [match.group().encode() for match in iters]
    return bytes_words


def pre_tokenize_with_special(
        chunk: str,
        special_tokens: list[str] = None
) -> list[bytes]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if special_tokens is None:
        iters = re.finditer(PAT, chunk)
        bytes_words = [match.group().encode() for match in iters]
        return bytes_words

    escaped = sorted((re.escape(tok) for tok in special_tokens), key=len, reverse=True)
    special_pat = "(" + "|".join(escaped) + ")"

    parts = re.split(special_pat, chunk)
    bytes_words: list[bytes] = []
    for part in parts:
        if not part or part in special_tokens:
            continue

        iters = re.finditer(PAT, part)
        for match in iters:
            bytes_words.append(match.group().encode())
    return bytes_words


def split_words_with_special(
        chunk: str,
        special_tokens: list[str] = None
) -> list[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if special_tokens is None:
        iters = re.finditer(PAT, chunk)
        bytes_words = [match.group() for match in iters]
        return bytes_words

    escaped = sorted((re.escape(tok) for tok in special_tokens), key=len, reverse=True)
    special_pat = "(" + "|".join(escaped) + ")"
    parts = re.split(special_pat, chunk)
    bytes_words: list[str] = []
    for part in parts:
        if not part:
            continue

        if part in special_tokens:
            bytes_words.append(part)
            continue

        iters = re.finditer(PAT, part)
        for match in iters:
            bytes_words.append(match.group())
    return bytes_words


def gpt2_byte_to_unicode():
    """
    extract from https://huggingface.co/transformers/v2.9.1/_modules/transformers/tokenization_gpt2.html

    Returns list of utf-8 byte and a mapping to unicode strings.
    We specifically avoids mapping to whitespace/control characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def save_vocab_merges(vocab: dict,
                      merges,
                      vocab_path,
                      merges_path):
    byte_to_unicode = gpt2_byte_to_unicode()

    inv_vocab = {''.join(byte_to_unicode[byte_token] for byte_token in bytes_token): idx
                 for idx, bytes_token in vocab.items()}
    with open(vocab_path, 'w', encoding="utf-8") as f:
        json.dump(inv_vocab, f, ensure_ascii=False, indent=2)
    f.close()

    inv_merges = [
        (''.join(byte_to_unicode[byte_token] for byte_token in bytes_token1),
         ''.join(byte_to_unicode[byte_token] for byte_token in bytes_token2)
         )
        for bytes_token1, bytes_token2 in merges
    ]
    with open(merges_path, 'w', encoding="utf-8") as f:
        for merge1, merge2 in inv_merges:
            f.write(merge1 + " " + merge2 + "\n")
    f.close()


def load_vocab_merges(vocab_path, merges_path):
    unicode_to_byte = {v: k for k, v in gpt2_byte_to_unicode().items()}

    with open(merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        merges = [
            (
                bytes([unicode_to_byte[token] for token in merge_token_1]),
                bytes([unicode_to_byte[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]

    # Compare the vocab to the expected output vocab
    with open(vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        vocab = {
            gpt2_vocab_index: bytes([unicode_to_byte[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }

    return vocab, merges


if __name__ == "__main__":
    text = """Once upon a
<|endoftext|>
there was a reliable otter named Ollie.
<|endoftext|>
One day, a little"""

    special_tokens = ['<|endoftext|>']

    words = split_words_with_special(text, special_tokens)
    print(words)