"""Paths and model versions shared by the API, inference and downloader."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BACKEND = "marian_ct2"
LEGACY_TOKENIZER = "Helsinki-NLP/opus-mt-en-vi"
LEGACY_MODELS = {"vi_en": "vienv2.pt", "en_vi": "enviv4.pt"}
CACHE_CAPACITY = 500

MODEL_SPECS = {
    "vi_en": {
        "repo_id": "HungNguyen29/MarianNMT-vi2en-ct2-int8",
        "revision": "85576987efd66124a14442c3f4499723b1fb852a",
        "tokenizer_repo": "HungNguyen29/MarianNMT-PhoMT_vi2en",
        "tokenizer_revision": "2bded11702bdb31afa5f352aa4b8d4258719906a",
    },
    "en_vi": {
        "repo_id": "HungNguyen29/MarianNMT-en2vi-ct2-int8",
        "revision": "de5e98f400517cef690ca16a98e3e2cdc1734b5c",
        "tokenizer_repo": "HungNguyen29/MarianNMT-PhoMT_en2vi",
        "tokenizer_revision": "3c4e896ab4e046ca0a9c024a201ec39c53e2f616",
    },
}
CACHE_DIR = Path(os.environ.get("VIE2ENG_MODEL_CACHE", BASE_DIR / "models"))
