import mmap
import cProfile
from collections import defaultdict
import time
from tqdm import tqdm
from multiprocessing import Pool


from cs336_basics.tokenization.prepare import find_chunk_boundaries, pre_tokenize_with_special, save_vocab_merges, load_vocab_merges


def chunk_infos_count(chunk: str, special_tokens: list[str]):
    bytes_word_counts = defaultdict(int)  # {b"I": 1}
    pair_counts = defaultdict(int)  # {(0, 1): 3, (1,2): 4}
    indices_bytes_words = {}
    pair_in_words = defaultdict(set)

    bytes_words = pre_tokenize_with_special(chunk, special_tokens)  # split text to words: [b'I', b"'m", b' doing']
    for bytes_word in bytes_words:
        bytes_word_counts[bytes_word] += 1

    # {b'I': [200], b"'m": [100, 120]}
    for bytes_word in bytes_word_counts.keys():
        indices_bytes_words[bytes_word] = list(map(int, bytes_word))
        indices_bytes_word = indices_bytes_words[bytes_word]

        for pair in zip(indices_bytes_word, indices_bytes_word[1:]):
            pair_counts[pair] += bytes_word_counts[bytes_word]
            pair_in_words[pair].add(bytes_word)

    return pair_counts, bytes_word_counts, indices_bytes_words, pair_in_words


def vocab_merge_process(vocab_max_size: int,
                        special_tokens: list[str],
                        pair_counts: dict,
                        bytes_word_counts: dict,
                        indices_bytes_words: dict,
                        pair_in_words: dict):
    vocab_size = 256  # initial size with Only 256 bytes
    vocab = {i: bytes([i]) for i in range(256)}

    for special_token in special_tokens:  # special tokens with ids which begin at 256
        vocab[vocab_size] = special_token.encode()
        vocab_size += 1

    merges: list[tuple[bytes, bytes]] = []
    indices_merges: list[tuple[int, int]] = []

    for i in tqdm(range(vocab_size, vocab_max_size), desc="BPE training"):
        pair_max_single = max(pair_counts, key=pair_counts.get)
        pairs_max = [pair for pair, pair_count in pair_counts.items() if pair_count == pair_counts[pair_max_single]]
        # break ties in pair frequency by preferring the lexicographically greater pair.
        pair_max = max(pairs_max, key=lambda p: (vocab[p[0]], vocab[p[1]]))
        # print(pair_max, pair_counts[pair_max], len(pair_counts))

        # merge
        index1, index2 = pair_max
        new_index = vocab_size

        vocab[new_index] = vocab[index1] + vocab[index2]
        merges.append((vocab[index1], vocab[index2]))
        indices_merges.append((index1, index2))
        vocab_size += 1

        affected_bytes_words = list(pair_in_words[pair_max])
        for bytes_word in affected_bytes_words:
            indices = indices_bytes_words[bytes_word]
        # for bytes_word, indices in indices_bytes_words.items():
            new_indices = []
            j = 0
            while j < len(indices):
                if j + 1 < len(indices) and indices[j] == index1 and indices[j + 1] == index2:
                    new_indices.append(new_index)

                    if j >= 1:
                        pair_counts[(indices[j - 1], indices[j])] -= bytes_word_counts[bytes_word]
                        if pair_counts[(indices[j - 1], indices[j])] <= 0:
                            pair_in_words[(indices[j - 1], indices[j])].discard(bytes_word)

                    if j + 2 < len(indices):
                        pair_counts[(indices[j + 1], indices[j + 2])] -= bytes_word_counts[bytes_word]
                        if pair_counts[(indices[j + 1], indices[j + 2])] <= 0:
                            pair_in_words[(indices[j + 1], indices[j + 2])].discard(bytes_word)

                    pair_counts[pair_max] -= bytes_word_counts[bytes_word]
                    if pair_counts[pair_max] <= 0:
                        del pair_counts[pair_max]
                        pair_in_words[pair_max].discard(bytes_word)

                    j += 2
                else:
                    new_indices.append(indices[j])
                    j += 1

            for idx1, idx2 in zip(new_indices[:], new_indices[1:]):
                if idx1 == new_index or idx2 == new_index:
                    pair_counts[(idx1, idx2)] += bytes_word_counts[bytes_word]
                    pair_in_words[(idx1, idx2)].add(bytes_word)

            indices_bytes_words[bytes_word] = new_indices

        assert pair_counts[pair_max] == 0
    return vocab, merges, indices_merges


def train_bpe(
        input_path: str,
        vocab_max_size: int,
        special_tokens: list[str]
):
    """
    incrementally updating vocabulary
    """
    with open(input_path, "rb") as f:
        chunk = f.read().decode("utf-8", errors="ignore")  # text: "I'm doing this"
        f.close()

    pair_counts, bytes_word_counts, indices_bytes_words, pair_in_words = chunk_infos_count(chunk, special_tokens)

    return vocab_merge_process(vocab_max_size, special_tokens, pair_counts, bytes_word_counts, indices_bytes_words, pair_in_words)


def process_chunk(args):
    input_path, boundary_st, boundary_end, special_tokens = args

    with open(input_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
        chunk = mm[boundary_st: boundary_end].decode("utf-8", errors="ignore")
        mm.close()
    f.close()

    return chunk_infos_count(chunk, special_tokens)


def train_bpe_parallelism(
        input_path: str,
        vocab_max_size: int,
        special_tokens: list[str],
        num_chunks: int,
        num_processes: int
):
    """
    incrementally updating vocabulary
    """

    bytes_word_counts = defaultdict(int)  # {b"I": 1}
    pair_counts = defaultdict(int)  # {(0, 1): 3, (1,2): 4}
    indices_bytes_words = {}
    pair_in_words = defaultdict(set)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        f.close()

    num_chunks = len(boundaries) - 1
    print("boundaries: ", boundaries, "num_chunks: ", num_chunks)
    with Pool(processes=num_processes) as pool:  # 8个进程，可根据CPU核心数调整
        args = [(input_path, boundaries[i], boundaries[i + 1], special_tokens) for i in range(num_chunks)]
        results = tqdm(pool.imap_unordered(process_chunk, args),
                       total=num_chunks, desc="BPE chunks")


        for result in results:
            local_pairs, local_bw_counts, local_indices, local_pair_words = result

            for pair, cnt in local_pairs.items():
                pair_counts[pair] += cnt

            for bytes_word, cnt in local_bw_counts.items():
                bytes_word_counts[bytes_word] += cnt

            for bytes_word, indices_word in local_indices.items():
                indices_bytes_words[bytes_word] = indices_word

            for pair, words in local_pair_words.items():
                pair_in_words[pair].update(words)

    return vocab_merge_process(vocab_max_size, special_tokens, pair_counts, bytes_word_counts, indices_bytes_words, pair_in_words)



if __name__ == "__main__":
    import pathlib

    File_PATH = pathlib.Path(__file__).resolve().parent

    start = time.perf_counter()

    text_file = "TinyStories_valid"
    input_path = f"data/TinyStories/TinyStoriesV2-GPT4-valid.txt"
    # input_path = f"data/owt/{text_file}.txt"

    # input_path = f"../../data/TinyStories/"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    # vocab, merges, indices_merges = train_bpe(input_path, vocab_size, special_tokens)
    # pr = cProfile.Profile()
    # pr.enable()

    # vocab, merges, indices_merges = train_bpe(input_path, vocab_size, special_tokens)
    vocab, merges, indices_merges = train_bpe_parallelism(input_path, vocab_size, special_tokens, 20, 20)


    # pr.disable()
    # pr.print_stats(sort='cumulative')  # 显示前20行

    longest_token = b''
    longest_idx = -1
    for idx, bytes_token in vocab.items():
        if len(longest_token) < len(bytes_token):
            longest_token = bytes_token
            longest_idx = idx
    print("longest token: ", longest_idx, " ", vocab[longest_idx])

    save_vocab_merges(vocab, merges,
                      f"{File_PATH}/{text_file}_vocab.json",
                      f"{File_PATH}/{text_file}_vocab_merges.txt"
                      )
    end = time.perf_counter()
    print(f"train bpe : {end - start:.2f}s")

    ref_vocab, ref_merges = load_vocab_merges(f"{File_PATH}/{text_file}_vocab.json",f"{File_PATH}/{text_file}_vocab_merges.txt")

    assert len(merges) == len(ref_merges)
    assert merges == ref_merges
    assert len(vocab) == len(ref_vocab)
    assert set(vocab.keys()) == set(ref_vocab.keys())
    assert set(vocab.values()) == set(ref_vocab.values())