"""Run inference on one or more PIL images and produce aggregated output."""
import torch

from app.inference.postprocess import aggregate_raw_levels, build_response
from app.inference.transforms import prepare_tensor
from app.model import get_model


def predict_image(pil_img) -> dict:
    model = get_model()

    img_tensor = prepare_tensor(pil_img)
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    pred_logits = outputs["pred_logits"][0].cpu()
    pred_boxes = outputs["pred_boxes"][0].cpu()
    pred_attrs = outputs["pred_attributes"][0].cpu()

    probas = _select_probas(pred_logits)
    selected_index = _select_best_index(probas)

    return build_response(probas, pred_boxes, pred_attrs, selected_index, pil_img.size)


def predict_images(pil_images: list) -> dict:
    """Aggregate predictions across multiple images of the same avocado.

    Runs each image through the model, keeps those with a plausible detection,
    and combines their raw ripeness by median.
    """
    per_image = [predict_image(img) for img in pil_images]

    raw_levels = []
    per_image_results = []
    for res in per_image:
        if res.get("confidence", 0) <= 0.0:
            continue
        raw_levels.append(res["raw_level"])
        per_image_results.append(res)

    if not raw_levels:
        return {
            "ripeness": None,
            "confidence": 0.0,
            "advice": "No avocado detected in any photo. Please retake with clearer images.",
            "raw_level": None,
            "selected_attr": 0.0,
            "max_attr": 0.0,
            "box": None,
            "images_used": 0,
            "per_image": per_image_results,
        }

    aggregated = aggregate_raw_levels(raw_levels)
    # Reuse the confidence/box from the image closest to the median (most representative).
    representative = min(per_image_results, key=lambda r: abs(r["raw_level"] - aggregated))

    return {
        "ripeness": representative["ripeness"],
        "confidence": representative["confidence"],
        "advice": representative["advice"],
        "raw_level": round(aggregated, 2),
        "selected_attr": representative["selected_attr"],
        "max_attr": representative["max_attr"],
        "box": representative["box"],
        "images_used": len(raw_levels),
        "per_image": per_image_results,
    }


def _select_probas(pred_logits: torch.Tensor) -> torch.Tensor:
    probas_0 = pred_logits.sigmoid()[:, 0]
    if pred_logits.shape[1] > 1:
        probas_1 = pred_logits.sigmoid()[:, 1]
        return probas_1 if probas_1.max() > probas_0.max() else probas_0
    return probas_0


def _select_best_index(probas: torch.Tensor) -> int:
    keep = probas > 0.5
    if keep.sum() == 0:
        keep = torch.zeros_like(probas, dtype=torch.bool)
        keep[probas.argmax()] = True
    return torch.where(keep)[0][0].item()
