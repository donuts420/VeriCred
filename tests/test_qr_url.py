import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image


def test_extract_urls_from_text():
    from qr_validation import extract_urls
    urls = extract_urls("verify at https://example.edu/verify/123 and http://x.co/a")
    assert "https://example.edu/verify/123" in urls


def test_validate_syntax():
    from qr_validation import validate_url
    ok = validate_url("https://example.edu/verify/123", check_reachable=False)
    assert ok["present"] and ok["valid_syntax"] and ok["domain"] == "example.edu"
    bad = validate_url("not a url", check_reachable=False)
    assert bad["present"] and not bad["valid_syntax"]


def test_blank_image_has_no_qr():
    from qr_validation import detect_qr
    res = detect_qr(Image.new("RGB", (100, 100), "white"))
    assert res["found"] is False


def test_missing_qr_is_neutral():
    from qr_validation import analyze
    res = analyze(Image.new("RGB", (100, 100), "white"), "plain text",
                  check_reachable=False)
    assert res["qr_found"] is False
    assert res["suspicious"] is False
