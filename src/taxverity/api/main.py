"""The process entrypoint uvicorn (and Lambda Web Adapter, which proxies to it) import."""

from taxverity.api.app import create_app

app = create_app()
