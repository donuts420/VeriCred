import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fitz  # PyMuPDF


def _make_text_pdf(path, text):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_import_has_no_side_effects():
    # Would raise EOFError before the __main__ guard (input() at import)
    import ocr.extractor  # noqa: F401


def test_extract_certificate_text_pdf(tmp_path):
    from ocr.extractor import extract_certificate
    p = tmp_path / "cert.pdf"
    _make_text_pdf(str(p),
        "This is to certify that Alice Example\n"
        "Bachelor of Technology\n"
        "Example University\n"
        "Year 2023\n"
        "Certificate No: CERT-12345")
    data = extract_certificate(str(p))
    assert "Alice Example" in data["raw_text"]


def test_extract_certificate_rejects_bad_extension(tmp_path):
    from ocr.extractor import extract_certificate
    try:
        extract_certificate(str(tmp_path / "x.txt"))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_extract_fields_never_invents():
    from ocr.extractor import extract_fields
    assert extract_fields("") == {
        "student_name": None, "degree": None, "institution": None,
        "year": None, "certificate_number": None,
    }


def test_extract_fields_finds_present_fields():
    from ocr.extractor import extract_fields
    f = extract_fields(
        "This is to certify that Alice Example\n"
        "Bachelor of Technology\nExample University\n"
        "Year 2023\nCertificate No: CERT-12345")
    assert f["student_name"] == "Alice Example"
    assert "Bachelor" in f["degree"]
    assert "University" in f["institution"]
    assert f["year"] == "2023"
    assert f["certificate_number"] == "CERT-12345"


def test_image_ocr_regression_certificate(tmp_path):
    """Direct tesseract gives ~305 chars; pipeline must be comparable, not ~100 garbage."""
    from ocr.extractor import extract_certificate
    src = Path("d1Certificate_forgery_detection/test/images/"
               "f4_png.rf.83af655c4721072cda4d4c02798c0f9a.jpg")
    if not src.exists():
        return  # dataset not present (e.g. CI); skip
    text = extract_certificate(str(src))["raw_text"] or ""
    assert len(text.strip()) >= 200, f"only {len(text)} chars: {text[:100]!r}"
