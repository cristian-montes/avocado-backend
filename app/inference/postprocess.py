"""Post-processing: map raw model outputs to a ripeness prediction."""
import statistics

import torch


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)


def normalized_box_to_pixels(box: torch.Tensor, image_width: int, image_height: int) -> dict:
    x0, y0, x1, y1 = cxcywh_to_xyxy(box).clamp(0.0, 1.0).tolist()
    return {
        "x": round(x0 * image_width),
        "y": round(y0 * image_height),
        "width": round((x1 - x0) * image_width),
        "height": round((y1 - y0) * image_height),
    }


def build_response(
    probas: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_attrs: torch.Tensor,
    selected_index: int,
    image_size: tuple[int, int],
) -> dict:
    """Turn selected raw outputs into the API response dict."""
    selected_box = pred_boxes[selected_index]
    selected_attr = float(pred_attrs[selected_index, 0])
    final_confidence = float(probas[selected_index])

    pooled_attr = max(0.0, min(1.0, selected_attr))
    ripeness_score = (pooled_attr * 4.0) + 1.0
    avocado_box = normalized_box_to_pixels(selected_box, *image_size)

    if ripeness_score < 1.8:
        ripeness_label = "Firm & Green (Stage 1 - Unripe)"
        advice = "Rock hard. Needs about 4-5 days. Keep it at room temperature in a paper bag with a banana to accelerate ripening."
    elif 1.8 <= ripeness_score < 2.5:
        ripeness_label = "Breaking (Stage 2 - Turning)"
        advice = "Slightly yielding at the stem. Needs 2 more days. Ideal if you are prepping sliced elements for later in the week."
    elif 2.5 <= ripeness_score < 3.6:
        ripeness_label = "Perfectly Ripe (Stage 3 - Ready!)"
        advice = (
            "Optimal yielding texture. Ideal for guacamole or spreading on avocado "
            "toast right now. Move to the fridge to preserve this state."
        )
    elif 3.6 <= ripeness_score < 4.3:
        ripeness_label = "Firm-Safe Overripe (Stage 4 - Eat Immediately)"
        advice = "Very soft. Yields deeply to slight pressure. Check internal flesh carefully for minor bruising, perfect for immediate smoothies or dressings."
    else:
        ripeness_label = "Past Prime (Stage 5 - Spoiled)"
        advice = "Significantly hollow or mushy feel. High likelihood of structural browning inside. Inspect structural integrity before using."

    return {
        "ripeness": ripeness_label,
        "confidence": round(final_confidence, 2),
        "advice": advice,
        "raw_level": round(ripeness_score, 2),
        "selected_attr": round(selected_attr, 4),
        "max_attr": round(float(pred_attrs.max()), 4),
        "box": avocado_box,
    }


def aggregate_raw_levels(raw_levels: list[float]) -> float:
    """Combine per-image ripeness scores using the median.

    Median resists a single bad angle/photo skewing the result, unlike a mean.
    """
    if not raw_levels:
        raise ValueError("No detections to aggregate")
    return statistics.median(raw_levels)
