"""Run with python server.py or uvicorn server:app."""

from vie2eng.api import create_app

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
