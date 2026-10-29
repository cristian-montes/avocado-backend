"""Centralized configuration loaded from environment variables."""
import os

from PIL import Image

# Upload / input limits
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

# CORS
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

# Clerk / JWT
JWKS_URL = os.getenv("CLERK_JWKS_URL")
JWT_ISSUER = os.getenv("CLERK_JWT_ISSUER")
JWT_AUDIENCE = os.getenv("CLERK_JWT_AUDIENCE")

# Model artifact
MODEL_SHA256 = os.getenv("MODEL_SHA256")
MODEL_GCS_URI = os.getenv("MODEL_GCS_URI")
MODEL_PATH = os.getenv("MODEL_PATH", "/tmp/model.pth")
RFDETR_DOWNLOAD_DIR = os.getenv("RFDETR_DOWNLOAD_DIR", "/tmp/rfdetr")

# Rate limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))
