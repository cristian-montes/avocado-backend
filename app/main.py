"""FastAPI application entrypoint."""
import io
import logging
from typing import List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from app import config
from app.auth import verify_clerk_user
from app.inference import predict_images
from app.inference.transforms import strip_exif
from app.model import model_is_loaded
from app.rate_limit import limiter

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_REQUEST = 4

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/")
def home():
    return {
        "status": "Avocado API is running",
        "model_loaded": model_is_loaded(),
    }


def _decode_image(file: UploadFile) -> Image.Image:
    """Validate and decode a single upload into a clean RGB PIL image."""
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Unsupported image type")

    contents = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(contents) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large")

    with Image.open(io.BytesIO(contents)) as image:
        image.verify()
    with Image.open(io.BytesIO(contents)) as image:
        return strip_exif(image)


@app.post("/predict")
async def predict(
    files: List[UploadFile] = File(...),
    user_id: str = Depends(verify_clerk_user),
):
    limiter.check(user_id)
    if len(files) > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Maximum {MAX_IMAGES_PER_REQUEST} images per request",
        )

    try:
        pil_images = [_decode_image(f) for f in files]
        return predict_images(pil_images)
    except HTTPException:
        raise
    except (DecompressionBombError, UnidentifiedImageError, OSError) as e:
        logger.info("Rejected invalid image upload: %s", type(e).__name__)
        raise HTTPException(status_code=400, detail="Invalid image") from e
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed") from e
