"""Validated API request and response contracts."""

from typing import Literal

from pydantic import BaseModel, Field

from .config import DEFAULT_BACKEND


class TranslationRequest(BaseModel):
    text: str
    source_lang: Literal["auto", "vi", "en"] = "auto"
    target_lang: Literal["auto", "vi", "en"] = "auto"
    model_backend: Literal["marian_ct2", "legacy"] = DEFAULT_BACKEND
    method: Literal["beam", "greedy"] = "beam"
    beam_size: int = Field(default=4, ge=1, le=6)
    max_length: int = Field(default=60, ge=2, le=128)


class TranslationResponse(BaseModel):
    translated_text: str
    detected_lang: str
    source_lang: str
    target_lang: str
    model_name: str
    latency_ms: float
    method: str
    from_cache: bool
