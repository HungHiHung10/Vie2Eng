"""FastAPI composition and HTTP endpoints."""

import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import BASE_DIR
from .schemas import TranslationRequest, TranslationResponse
from .service import TranslationUnavailable, create_translation_service


def _asset_version():
    digest = hashlib.sha256()
    for path in sorted((BASE_DIR / "static").rglob("*.js")):
        digest.update(path.relative_to(BASE_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def create_app(service=None):
    service = service if service is not None else create_translation_service()
    app = FastAPI(
        title="Google Translate - PhoMT Marian NMT",
        description="Dual Transformer Translation System for Vietnamese and English",
        version="1.0.0",
    )
    app.state.translation_service = service
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.middleware("http")
    async def refresh_client_assets(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.endswith((".js", ".css")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def get_index():
        html = (BASE_DIR / "templates/index.html").read_text(encoding="utf-8")
        return HTMLResponse(html.replace("__ASSET_VERSION__", _asset_version()))

    @app.get("/logo.png", response_class=FileResponse)
    def get_logo():
        return FileResponse(BASE_DIR / "static/logo.png")

    @app.get("/api/health")
    def health_check():
        return service.health()

    @app.post("/api/translate", response_model=TranslationResponse)
    def translate(request: TranslationRequest):
        try:
            return service.translate(request)
        except TranslationUnavailable:
            raise HTTPException(status_code=503, detail=(
                "Không thể tải hoặc chạy model. Chạy python download_models.py để chuẩn bị "
                "model MarianNMT; kiểm tra checkpoint .pt nếu dùng PhoMT cũ."
            )) from None

    return app
