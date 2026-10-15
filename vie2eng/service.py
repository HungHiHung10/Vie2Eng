"""Translation orchestration, independent of HTTP and model implementations."""

import logging
from time import perf_counter

from .cache import TranslationCache
from .config import DEFAULT_BACKEND, LEGACY_MODELS, MODEL_SPECS
from .language import detect_language
from .schemas import TranslationRequest, TranslationResponse

logger = logging.getLogger(__name__)


class TranslationUnavailable(Exception):
    """A backend could not prepare or run its model."""


class TranslationService:
    def __init__(self, backends, cache=None):
        self.backends = backends
        self.cache = cache if cache is not None else TranslationCache()

    def health(self):
        return {
            "status": "healthy",
            "device": str(self.backends["legacy"].device),
            "models": {direction: spec["repo_id"] for direction, spec in MODEL_SPECS.items()},
            "default_backend": DEFAULT_BACKEND,
            "ct2_device": str(self.backends["marian_ct2"].device),
            "legacy_models": LEGACY_MODELS,
            "cache_size": len(self.cache),
        }

    def translate(self, request: TranslationRequest) -> TranslationResponse:
        text = request.text.strip()
        if not text:
            return TranslationResponse(
                translated_text="", detected_lang="vi", source_lang=request.source_lang,
                target_lang=request.target_lang, model_name="none", latency_ms=0.0,
                method=request.method, from_cache=False,
            )

        started = perf_counter()
        detected = detect_language(text)
        source = detected if request.source_lang == "auto" else request.source_lang
        target = request.target_lang
        if target == "auto" or target == source:
            target = "en" if source == "vi" else "vi"
        direction = "vi_en" if source == "vi" else "en_vi"
        model_name = self._model_name(request.model_backend, direction)
        method = "greedy" if request.method == "greedy" or request.beam_size == 1 else "beam"
        beam_size = request.beam_size if method == "beam" else 1
        key = (model_name, source, target, method, beam_size, request.max_length, text)

        translated = self.cache.get(key)
        from_cache = translated is not None
        if not from_cache:
            try:
                translated = self.backends[request.model_backend].translate(
                    text, direction, method, beam_size, request.max_length,
                )
            except Exception as exc:
                # Exception messages from download libraries may contain credentials.
                logger.error("Model %s failed (%s)", model_name, type(exc).__name__)
                raise TranslationUnavailable from None
            self.cache.put(key, translated)

        return TranslationResponse(
            translated_text=translated, detected_lang=detected,
            source_lang=source, target_lang=target, model_name=model_name,
            latency_ms=round((perf_counter() - started) * 1000, 2),
            method=method, from_cache=from_cache,
        )

    @staticmethod
    def _model_name(backend, direction):
        if backend == "marian_ct2":
            return MODEL_SPECS[direction]["repo_id"]
        label = "Vi -> En" if direction == "vi_en" else "En -> Vi"
        return f"{LEGACY_MODELS[direction]} (PhoMT {label})"


def create_translation_service():
    # Models and tokenizers stay unloaded until the first translation request.
    from .backends.ct2 import CT2Backend
    from .backends.legacy import LegacyBackend

    return TranslationService({"marian_ct2": CT2Backend(), "legacy": LegacyBackend()})
