from typing import Iterable, Iterator
import json
import regex as re

from cs336_basics.tokenization.prepare import pre_tokenize, find_chunk_boundaries,split_words_with_special


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

    def _load_vocab_json(self):
        self.vocab_bytes_id = {
            vocab_item: vocab_index
            for vocab_index, vocab_item in self.vocab.items()
        }

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
                # print(bytes_words)
                for i in range(len(bytes_words)):
                    bytes_word: bytes = bytes_words[i]
                    split_bytes_word: list[bytes] = [bytes_word[i:i + 1] for i in range(len(bytes_word))]
                    new_bytes_word: list[bytes] = []
                    for bytes1_merge, bytes2_merge in self.merges:
                        j = 0
                        while j < len(split_bytes_word):
                            if j + 1 < len(split_bytes_word) and split_bytes_word[j] == bytes1_merge and \
                                    split_bytes_word[j + 1] == bytes2_merge:
                                new_bytes_word.append(bytes1_merge + bytes2_merge)
                                j += 2
                            else:
                                new_bytes_word.append(split_bytes_word[j])
                                j += 1
                        split_bytes_word = new_bytes_word
                        new_bytes_word = []

                    # tokens to IDs
                    # print(split_bytes_word)
                    for split_bytes in split_bytes_word:
                        text_indices.append(self.vocab_bytes_id[split_bytes])

        return text_indices

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for item in iterable:
            words = split_words_with_special(item, self.special_tokens)
            for word in words:
                yield from self.encode(word)

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""

        bytes_text: bytes = bytearray()
        for item in ids:
            bytes_text += self.vocab[item]

        return bytes_text.decode('utf-8', errors="replace")


if __name__ == "__main__":
    # tokenizer = Tokenizer.from_files(f"../tests/fixtures/gpt2_vocab.json",
    #                                  f"../tests/fixtures/gpt2_merges.txt",
    #                                  ["<|endoftext|>"])
    tokenizer = Tokenizer.from_files(f"../../tests/fixtures/train-bpe-reference-vocab.json",
                                     f"../tests/fixtures/train-bpe-reference-merges.txt",
                                     ["<|endoftext|>"])
    text = "Hello, world!"
    encode_text = tokenizer.encode(text)
    decode_text = tokenizer.decode(encode_text)
    print(encode_text)
    print(decode_text)
