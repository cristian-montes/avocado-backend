"""Image preprocessing transforms (must match the fine-tuning notebook)."""
import torch
import torchvision.transforms as T
from PIL import Image

transform = T.Compose(
    [
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def prepare_tensor(pil_img) -> torch.Tensor:
    return transform(pil_img).unsqueeze(0)


def strip_exif(image: Image.Image) -> Image.Image:
    """Return a copy of the image with all metadata (EXIF/GPS) removed.

    `convert("RGB")` builds a fresh pixel-only image; anything not part of the
    pixel data (GPS, timestamp, device, thumbnail) is discarded.
    """
    return image.convert("RGB")
