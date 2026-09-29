import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _evidence():
    return {
        "fields": {"student_name": None, "degree": None, "institution": None,
                   "year": None, "certificate_number": None},
        "ocr_chars": 0,
        "det": {"model_loaded": True, "boxes": [
            {"label": "fake", "confidence": 0.8, "bbox": [1, 2, 3, 4]}],
            "fake_count": 1, "true_count": 0, "max_fake_conf": 0.8},
        "qr": {"qr_found": False, "urls": []},
        "decision": {"status": "Potentially Tampered", "reasons": ["r1"]},
    }


def test_template_works_without_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    from genai.explain import explain
    out = explain(_evidence())
    assert "Potentially Tampered" in out
    assert "0.80" in out and "sha" not in out.lower()


def test_never_claims_authenticity():
    from genai.explain import explain
    out = explain(_evidence()).lower()
    assert "authentic" not in out and "genuine" not in out
    assert "not proof" in out
