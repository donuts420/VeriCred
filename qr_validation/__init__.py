"""QR detection (OpenCV) + URL extraction/validation. Missing QR/URL is neutral."""
import re
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)


def detect_qr(image):
    """Decode QR from PIL image or path. Never raises on bad input."""
    try:
        if isinstance(image, (str, bytes)) or hasattr(image, "__fspath__"):
            image = Image.open(image)
        arr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        data, points, _ = cv2.QRCodeDetector().detectAndDecode(arr)
        if data:
            return {"found": True, "payload": data}
    except Exception:
        pass
    return {"found": False, "payload": None}


def extract_urls(text):
    return URL_RE.findall(text or "")


def validate_url(url, check_reachable=True, timeout=4):
    """Syntax always checked; reachability is best-effort (failure = unknown)."""
    out = {"url": url, "present": True, "valid_syntax": False,
           "domain": None, "reachable": None}
    try:
        p = urlparse(url)
        if p.scheme in ("http", "https") and p.netloc:
            out["valid_syntax"] = True
            out["domain"] = p.netloc
    except Exception:
        return out
    if out["valid_syntax"] and check_reachable:
        try:
            import requests
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            out["reachable"] = r.status_code < 400
        except Exception:
            out["reachable"] = None  # unknown, never suspicious by itself
    return out


def analyze(image, ocr_text, check_reachable=True):
    """Combine QR + text URLs. suspicious is always False (supporting evidence)."""
    qr = detect_qr(image)
    seen = list(dict.fromkeys(extract_urls(qr.get("payload"))
                              + extract_urls(ocr_text)))
    return {"qr_found": qr["found"], "qr_payload": qr["payload"],
            "urls": [validate_url(u, check_reachable) for u in seen],
            "suspicious": False}
