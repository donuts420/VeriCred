import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz


def test_app_imports():
    import app  # noqa: F401


def test_verify_file_end_to_end_no_weights(tmp_path, monkeypatch):
    import app
    monkeypatch.setenv("VERICRED_WEIGHTS", str(tmp_path / "missing.pt"))
    from detection import predict as P
    P._model, P._model_path = None, None
    doc = fitz.open()
    pg = doc.new_page()
    pg.insert_text((72, 72), "This is to certify that Bob Sample, Year 2022")
    pdf = tmp_path / "c.pdf"
    doc.save(pdf)
    doc.close()
    res = app.verify_file(pdf.read_bytes(), ".pdf", 0.5)
    assert "Bob Sample" in res["ocr"]["raw_text"]
    assert res["det"]["model_loaded"] is False
    assert res["decision"]["status"].startswith("Inconclusive")
    assert "sha256" not in res and "hash" not in res
    assert res["report"]
