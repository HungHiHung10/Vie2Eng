"""Local INT8 inference using the tokenizers from the original PhoMT models."""

import json
import os
import re
from pathlib import Path
from threading import Lock

import ctranslate2
from huggingface_hub import snapshot_download
from transformers import MarianTokenizer

from ..config import CACHE_DIR, MODEL_SPECS


def _snapshot(repo_id, revision, patterns):
    kwargs = dict(repo_id=repo_id, revision=revision, allow_patterns=patterns,
                  cache_dir=str(CACHE_DIR), token=os.environ.get("HF_TOKEN") or None)
    try:
        cached = snapshot_download(**kwargs, local_files_only=True)
        required = [name for name in patterns if name not in ("special_tokens_map.json", "added_tokens.json")]
        if all((Path(cached) / name).is_file() for name in required):
            return cached
    except FileNotFoundError:
        pass
    return snapshot_download(**kwargs)


def prepare_model(direction):
    """Download only CT2 weights and tokenizer files; reuse the cache offline."""
    spec = MODEL_SPECS[direction]
    model_path = _snapshot(spec["repo_id"], spec["revision"],
                           ["config.json", "model.bin", "shared_vocabulary.json"])
    tokenizer_path = _snapshot(spec["tokenizer_repo"], spec["tokenizer_revision"],
                               ["config.json", "tokenizer_config.json", "special_tokens_map.json",
                                "vocab.json", "source.spm", "target.spm", "added_tokens.json"])
    return model_path, tokenizer_path


class CT2Backend:
    device = "cpu"

    def __init__(self):
        self._loaded = {}
        self._load_lock = Lock()

    def get_model(self, direction):
        # A single load per direction, including concurrent first requests.
        with self._load_lock:
            if direction not in self._loaded:
                model_path, tokenizer_path = prepare_model(direction)
                tokenizer = MarianTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
                with open(Path(model_path) / "shared_vocabulary.json", encoding="utf-8") as stream:
                    vocabulary = json.load(stream)
                # The converted vocabulary excludes Marian's final <pad> token.
                if any(tokenizer.convert_ids_to_tokens(i) != token
                       for i, token in enumerate(vocabulary)):
                    raise ValueError(f"Tokenizer vocabulary does not match {MODEL_SPECS[direction]['repo_id']}")
                translator = ctranslate2.Translator(
                    model_path, device="cpu", compute_type="int8",
                    inter_threads=1, intra_threads=min(4, os.cpu_count() or 1),
                )
                self._loaded[direction] = (translator, tokenizer)
            return self._loaded[direction]

    def translate(self, text, direction, method="beam", beam_size=4, max_length=60):
        translator, tokenizer = self.get_model(direction)
        paragraphs = text.split("\n")
        chunks = []
        paragraph_indices = []
        # Split long sentences by tokens instead of silently truncating their tail.
        for index, paragraph in enumerate(paragraphs):
            for sentence in re.split(r"(?<=[.!?])\s+", paragraph.strip()):
                if not sentence:
                    continue
                ids = tokenizer.encode(sentence, add_special_tokens=False)
                for start in range(0, len(ids), max_length - 1):
                    source_ids = ids[start:start + max_length - 1] + [tokenizer.eos_token_id]
                    chunks.append(tokenizer.convert_ids_to_tokens(source_ids))
                    paragraph_indices.append(index)
        output = [[] for _ in paragraphs]
        if chunks:
            results = translator.translate_batch(
                chunks, beam_size=1 if method == "greedy" else beam_size,
                max_batch_size=16, max_input_length=max_length,
                max_decoding_length=max_length, min_decoding_length=1,
                no_repeat_ngram_size=3, length_penalty=0.6,
            )
            for index, result in zip(paragraph_indices, results):
                # Marian has distinct source and target SentencePiece processors.
                decoded = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(result.hypotheses[0]),
                    skip_special_tokens=True, use_source_tokenizer=False,
                ).strip()
                output[index].append(decoded)
        return "\n".join(" ".join(parts) for parts in output)
