import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_tampered_on_confident_fake():
    from analysis.decision import decide
    det = {"model_loaded": True, "boxes": [
        {"label": "fake", "confidence": 0.8, "bbox": [0, 0, 1, 1]}],
        "fake_count": 1, "true_count": 0, "max_fake_conf": 0.8}
    assert decide(det, {"suspicious": False}, 0.5)["status"] == "Potentially Tampered"


def test_weak_fake_below_threshold_not_tampered():
    from analysis.decision import decide
    det = {"model_loaded": True, "boxes": [
        {"label": "fake", "confidence": 0.2, "bbox": [0, 0, 1, 1]}],
        "fake_count": 1, "true_count": 2, "max_fake_conf": 0.2}
    r = decide(det, {"suspicious": False}, 0.5)
    assert r["status"] == "No high-confidence tampering detected"


def test_no_fake_is_not_proof_of_authentic():
    from analysis.decision import decide
    det = {"model_loaded": True, "boxes": [], "fake_count": 0,
           "true_count": 1, "max_fake_conf": 0.0}
    r = decide(det, {"suspicious": False}, 0.5)
    assert "authentic" not in r["status"].lower()
    assert any("not proof" in x.lower() for x in r["reasons"])


def test_unavailable_detector_is_inconclusive():
    from analysis.decision import decide
    det = {"model_loaded": False, "boxes": [], "fake_count": 0,
           "true_count": 0, "max_fake_conf": 0.0}
    r = decide(det, {"suspicious": False}, 0.5)
    assert r["status"].startswith("Inconclusive")


def test_missing_qr_never_tampered():
    from analysis.decision import decide
    det = {"model_loaded": True, "boxes": [], "fake_count": 0,
           "true_count": 0, "max_fake_conf": 0.0}
    r = decide(det, {"qr_found": False, "urls": [], "suspicious": False}, 0.5)
    assert r["status"] != "Potentially Tampered"


def test_decide_takes_no_hash():
    import inspect
    from analysis.decision import decide
    assert list(inspect.signature(decide).parameters) == ["det", "qr_url", "conf"]
