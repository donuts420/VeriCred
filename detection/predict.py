"""YOLO forgery-region detector. Lazy-loads weights; never invents boxes."""
import os
from pathlib import Path
from PIL import Image, ImageDraw

# Dataset classes: 0 = fake, 1 = true (see d1Certificate_forgery_detection/data.yaml)
CLASS_NAMES = {0: "fake", 1: "true"}

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "best.pt"

_model = None
_model_path = None


def resolve_weights(explicit=None):
    if explicit:
        return Path(explicit)
    env = os.environ.get("VERICRED_WEIGHTS")
    return Path(env) if env else DEFAULT_WEIGHTS


def _annotate(image, boxes):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        color = "red" if b["label"] == "fake" else "green"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 12)),
                  f"{b['label']} {b['confidence']:.2f}", fill=color)
    return out


def predict(image, conf=0.6, weights_path=None):
    """Run detection. image: PIL Image or path. Returns shared schema dict."""
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    wpath = resolve_weights(weights_path)

    def unavailable():
        return {"boxes": [], "fake_count": 0, "true_count": 0,
                "max_fake_conf": 0.0, "annotated": image,
                "model_loaded": False, "conf": conf,
                "weights": str(wpath)}

    if not wpath.is_file():
        return unavailable()
    try:
        from ultralytics import YOLO
    except ImportError:
        return unavailable()

    global _model, _model_path
    if _model is None or _model_path != str(wpath):
        _model = YOLO(str(wpath))
        _model_path = str(wpath)
    results = _model.predict(image, conf=conf, verbose=False)
    boxes = []
    for r in results:
        for box in r.boxes:
            cid = int(box.cls[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            boxes.append({"label": CLASS_NAMES.get(cid, str(cid)),
                          "confidence": float(box.conf[0]),
                          "bbox": [x1, y1, x2, y2]})
    fakes = [b["confidence"] for b in boxes if b["label"] == "fake"]
    return {"boxes": boxes,
            "fake_count": len(fakes),
            "true_count": len([b for b in boxes if b["label"] == "true"]),
            "max_fake_conf": max(fakes) if fakes else 0.0,
            "annotated": _annotate(image, boxes),
            "model_loaded": True, "conf": conf,
            "weights": str(wpath)}
