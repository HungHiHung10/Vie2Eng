import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from vie2eng.api import _asset_version, create_app
from vie2eng.cache import TranslationCache
from vie2eng.config import MODEL_SPECS
from vie2eng.service import TranslationService, create_translation_service


class TranslationApiTests(unittest.TestCase):
    def setUp(self):
        self.ct2 = SimpleNamespace(device="cpu", translate=Mock(return_value="Translated"))
        self.legacy = SimpleNamespace(device="cpu", translate=Mock(return_value="Translated"))
        self.service = TranslationService({"marian_ct2": self.ct2, "legacy": self.legacy})
        self.client = TestClient(create_app(self.service))

    def test_direction_and_default_backend(self):
        for text, direction in [("Tôi đang học tiếng Anh.", "vi_en"),
                                ("I am learning Vietnamese.", "en_vi")]:
            with self.subTest(direction=direction), patch.object(self.ct2, "translate", return_value="Translated") as translate:
                response = self.client.post("/api/translate", json={"text": text})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["model_name"], MODEL_SPECS[direction]["repo_id"])
                self.assertEqual(translate.call_args.args[1], direction)

    def test_cache_separates_length_and_backend(self):
        request = {"text": "Hello", "source_lang": "en"}
        with patch.object(self.ct2, "translate", return_value="Xin chào") as translate:
            first = self.client.post("/api/translate", json=request).json()
            second = self.client.post("/api/translate", json=request).json()
            longer = self.client.post("/api/translate", json={**request, "max_length": 100}).json()
            self.assertFalse(first["from_cache"])
            self.assertTrue(second["from_cache"])
            self.assertFalse(longer["from_cache"])
            self.assertEqual(translate.call_count, 2)
        with patch.object(self.legacy, "translate", return_value="Xin chào") as legacy:
            response = self.client.post("/api/translate", json={**request, "model_backend": "legacy"}).json()
            self.assertFalse(response["from_cache"])
            self.assertIn("enviv4.pt", response["model_name"])
            legacy.assert_called_once()

    def test_beam_one_reports_greedy_on_cached_requests(self):
        with patch.object(self.ct2, "translate", return_value="Xin chào"):
            request = {"text": "Hello", "beam_size": 1}
            for _ in range(2):
                response = self.client.post("/api/translate", json=request).json()
                self.assertEqual(response["method"], "greedy")

    def test_invalid_settings(self):
        for settings in ({"model_backend": "unknown"}, {"max_length": 0},
                         {"max_length": 513}, {"beam_size": 0}, {"source_lang": "fr"}):
            with self.subTest(settings=settings):
                response = self.client.post("/api/translate", json={"text": "Hello", **settings})
                self.assertEqual(response.status_code, 422)

    def test_model_failure_has_safe_actionable_error(self):
        with patch.object(self.ct2, "translate", side_effect=RuntimeError("private-token-value")):
            response = self.client.post("/api/translate", json={"text": "Hello"})
            self.assertEqual(response.status_code, 503)
            self.assertIn("download_models.py", response.json()["detail"])
            self.assertNotIn("private-token-value", response.text)
            self.assertFalse(self.service.cache)

    def test_empty_text_and_health_do_not_load_models(self):
        with patch.object(self.ct2, "translate") as translate, patch.object(self.legacy, "translate") as legacy:
            self.assertEqual(self.client.post("/api/translate", json={"text": "  "}).json()["translated_text"], "")
            self.assertEqual(self.client.get("/api/health").status_code, 200)
            translate.assert_not_called()
            legacy.assert_not_called()

    def test_model_selection_is_in_served_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="settingModel"', response.text)
        self.assertIn('value="marian_ct2"', response.text)
        self.assertIn('value="legacy"', response.text)

    def test_client_modules_and_page_revalidate(self):
        page = self.client.get("/")
        self.assertNotIn("__ASSET_VERSION__", page.text)
        script = re.search(r'<script type="module" src="([^"]+)"', page.text).group(1)
        self.assertRegex(script, r"app\.js\?v=[a-f0-9]{12}$")
        for url in ("/", script, "/static/js/languages.js", "/static/style.css"):
            with self.subTest(url=url):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["Cache-Control"], "no-cache")

    def test_empty_model_output_is_cached(self):
        self.ct2.translate.return_value = ""
        request = {"text": "Hello"}
        self.assertFalse(self.client.post("/api/translate", json=request).json()["from_cache"])
        self.assertTrue(self.client.post("/api/translate", json=request).json()["from_cache"])
        self.ct2.translate.assert_called_once()


class CacheAndAssetTests(unittest.TestCase):
    def test_cache_evicts_least_recently_used_entry(self):
        cache = TranslationCache(capacity=2)
        cache.put("a", "first")
        cache.put("b", "second")
        self.assertEqual(cache.get("a"), "first")
        cache.put("c", "third")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("a"), "first")
        self.assertEqual(len(cache), 2)

    def test_asset_version_changes_when_an_imported_module_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "static/js").mkdir(parents=True)
            (root / "static/app.js").write_text("import './js/languages.js';")
            module = root / "static/js/languages.js"
            module.write_text("export const value = 1;")
            with patch("vie2eng.api.BASE_DIR", root):
                before = _asset_version()
                self.assertEqual(before, _asset_version())
                module.write_text("export const value = 2;")
                self.assertNotEqual(before, _asset_version())


@unittest.skipUnless(os.environ.get("VIE2ENG_TEST_MODELS") == "1", "Requires downloaded models/checkpoints")
class LocalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = create_translation_service()
        cls.client = TestClient(create_app(cls.service))

    def test_real_models_both_directions_and_decoders(self):
        client = self.client
        for backend in ("marian_ct2", "legacy"):
            for source, text in (("vi", "Tôi đang học tiếng Anh."), ("en", "I am learning Vietnamese.")):
                for method in ("greedy", "beam"):
                    with self.subTest(backend=backend, source=source, method=method):
                        response = client.post("/api/translate", json={
                            "text": text, "source_lang": source,
                            "model_backend": backend, "method": method,
                        })
                        self.assertEqual(response.status_code, 200, response.text)
                        self.assertTrue(response.json()["translated_text"].strip())
                        if backend == "marian_ct2":
                            self.assertNotIn("<unk>", response.json()["translated_text"])
                        print(backend, source, method, ascii(response.json()["translated_text"]))

    def test_ct2_paragraphs_and_long_input(self):
        backend = self.service.backends["marian_ct2"]
        text = "I am learning Vietnamese.\n\n" + "Thank you very much " * 35 + "."
        translator, tokenizer = backend.get_model("en_vi")
        original_translate = translator.translate_batch
        batches = []

        class RecordingTranslator:
            def translate_batch(self, chunks, **kwargs):
                batches.extend(chunks)
                return original_translate(chunks, **kwargs)

        with patch.object(backend, "get_model", return_value=(RecordingTranslator(), tokenizer)):
            translated = backend.translate(text, "en_vi", max_length=20)
        self.assertEqual(translated.split("\n")[1], "")
        self.assertTrue(translated.split("\n")[2])
        self.assertTrue(all(len(chunk) <= 20 for chunk in batches))
        self.assertGreater(len(batches), 3)
        reconstructed = [token for chunk in batches[1:] for token in chunk[:-1]]
        expected = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(text.split("\n")[2], add_special_tokens=False)
        )
        self.assertEqual(reconstructed, expected)


if __name__ == "__main__":
    unittest.main()
