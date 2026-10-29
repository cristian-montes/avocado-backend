"""Thin entrypoint shim so `uvicorn main:app` resolves the packaged app.

All implementation lives under app/ (see app/main.py).
"""
from app.main import app

__all__ = ["app"]
