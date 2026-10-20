import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz


def test_app_imports():
    import app  # noqa: F401


def test_verify_file_end_to_end_no_weights(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "explain", lambda evidence: "Gemini report")
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


def test_gemini_error_is_shown_in_streamlit(monkeypatch):
    from types import SimpleNamespace
    import app
    from genai.explain import ReportGenerationError

    messages = []
    upload = SimpleNamespace(name="cert.pdf", getvalue=lambda: b"pdf")
    monkeypatch.setattr(app.st, "title", lambda *a: None)
    monkeypatch.setattr(app.st, "caption", lambda *a: None)
    monkeypatch.setattr(app.st, "file_uploader", lambda *a, **k: upload)
    monkeypatch.setattr(app.st, "slider", lambda *a, **k: 0.5)
    monkeypatch.setattr(app.st, "button", lambda *a, **k: True)
    monkeypatch.setattr(app.st, "error", messages.append)

    def fail(*args, **kwargs):
        raise ReportGenerationError("Gemini could not generate a complete report.")

    monkeypatch.setattr(app, "verify_file", fail)

    app.main()

    assert messages == ["Gemini could not generate a complete report."]
