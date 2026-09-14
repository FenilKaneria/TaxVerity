"""Step 15.1 — the process entrypoint uvicorn (and Lambda Web Adapter, which
proxies to it) import: `taxverity.api.main:app`. `create_app()` already owns
every startup decision (lifespan, middleware, routers); this module adds none
of its own."""

from taxverity.api.app import create_app

app = create_app()
