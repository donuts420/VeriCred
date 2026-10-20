import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _evidence():
    return {
        "fields": {"student_name": None, "degree": None, "institution": None,
                   "year": None, "certificate_number": None},
        "ocr_chars": 0,
        "det": {"model_loaded": True, "conf": 0.5, "boxes": [
            {"label": "fake", "confidence": 0.8, "bbox": [1, 2, 3, 4]}],
            "fake_count": 1, "true_count": 0, "max_fake_conf": 0.8},
        "qr": {"qr_found": False, "urls": []},
        "decision": {"status": "Potentially Tampered", "reasons": ["r1"]},
    }


def _model_response(explanation="Visual findings require careful review and do not prove fraud or legitimacy."):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
        {"explanation": explanation})))])


_DEFAULT_RESPONSE = object()


def _mock_openai(monkeypatch, response=_DEFAULT_RESPONSE, error=None):
    if response is _DEFAULT_RESPONSE:
        response = _model_response()
    captured = {}

    def create(**kwargs):
        captured["request"] = kwargs
        if error:
            raise error
        return response

    def openai_client(*, api_key, base_url):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=openai_client))
    return captured


@pytest.mark.parametrize("api_key", [None, ""])
def test_missing_or_empty_key_raises_clear_error(monkeypatch, api_key):
    if api_key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", api_key)
    from genai.explain import ReportGenerationError, explain
    with pytest.raises(ReportGenerationError, match="OPENROUTER_API_KEY"):
        explain(_evidence())


def test_openrouter_report_renders_facts_from_evidence(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test/model")
    from genai import explain as module
    evidence = _evidence()
    evidence["det"]["annotated"] = object()
    evidence["fields"]["student_name"] = "Alice Example"
    captured = _mock_openai(monkeypatch)

    report = module.explain(evidence)

    assert module._safe_report(report, evidence)
    for exact in ("Potentially Tampered", "Alice Example", "0 characters", "1 fake", "0 true",
                  "confidence 0.8", "[1, 2, 3, 4]", "QR code was not detected",
                  "No URLs detected", "r1", "Visual evidence does not prove fraud or legitimacy",
                  "No detected fake regions does not prove authenticity",
                  "Missing QR codes or URLs are neutral"):
        assert exact.lower() in report.lower()
    assert "detector is available" in report.lower()
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["request"]["model"] == "test/model"
    prompt = captured["request"]["messages"][0]["content"]
    sent = json.loads(prompt.split("Evidence JSON:\n", 1)[1])
    assert "annotated" not in sent["det"]
    assert sent["fields"]["student_name"] == "Alice Example"
    assert "qr" not in sent
    assert "https://verify.example" not in prompt


@pytest.mark.parametrize(("check", "mutate"), [
    ("numeric_evidence", lambda report: report.replace("confidence 0.8", "confidence high")),
    ("region_counts", lambda report: report.replace("1 fake", "2 fake")),
    ("ocr_character_count", lambda report: report.replace("0 characters", "characters unavailable")),
    ("qr_status", lambda report: report.replace("QR code was not detected", "QR code detected")),
    ("detector_availability", lambda report: report.replace(
        "Detector is available", "Detector is unavailable")),
])
def test_each_evidence_check_rejects_inconsistent_report(monkeypatch, check, mutate, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from genai import explain as module
    _mock_openai(monkeypatch)
    report = module.explain(_evidence())

    assert not module._safe_report(mutate(report), _evidence())
    assert check in caplog.text


def test_successful_report_preserves_extracted_fields_and_status(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from genai import explain as module
    evidence = _evidence()
    evidence["fields"]["student_name"] = "Alice Example"
    _mock_openai(monkeypatch)
    report = module.explain(evidence)
    assert "Potentially Tampered" in report
    assert "Student Name: Alice Example" in report


def test_openrouter_api_error_does_not_expose_key_in_error_or_logs(monkeypatch, caplog):
    key = "private-test-key"
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    from genai import explain as module
    _mock_openai(monkeypatch, error=RuntimeError(f"request failed with {key}"))

    with pytest.raises(module.ReportGenerationError, match="OpenRouter") as error:
        module.explain(_evidence())
    assert key not in str(error.value)
    assert key not in caplog.text


@pytest.mark.parametrize("content", [None, "", "   ", "not JSON", "{}", "[]", '{"explanation": 4}'])
def test_malformed_or_empty_openrouter_response_raises(monkeypatch, content):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from genai import explain as module
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    _mock_openai(monkeypatch, response=response)
    with pytest.raises(module.ReportGenerationError, match="OpenRouter"):
        module.explain(_evidence())


@pytest.mark.parametrize("response", [None, SimpleNamespace(), SimpleNamespace(choices=[]),
                                      SimpleNamespace(choices=[SimpleNamespace(
                                          message=SimpleNamespace(content=None))])])
def test_missing_openrouter_response_content_raises(monkeypatch, response):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from genai import explain as module
    _mock_openai(monkeypatch, response=response)
    with pytest.raises(module.ReportGenerationError, match="OpenRouter"):
        module.explain(_evidence())


def test_prompt_requests_wording_only_and_preserves_evidence_safeguards(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    captured = _mock_openai(monkeypatch)
    from genai import explain as module

    module.explain(_evidence())
    prompt = captured["request"]["messages"][0]["content"]
    for instruction in ("explanation", "Python will render those directly from the evidence",
                        "Do not repeat or infer any facts",
                        "Do not claim the certificate is authentic, genuine, or fraudulent"):
        assert instruction.lower() in prompt.lower()
    assert "0.8" in prompt and "[1, 2, 3, 4]" in prompt
    assert "Missing QR codes or URLs" not in prompt
    assert "qr" not in json.loads(prompt.split("Evidence JSON:\n", 1)[1])


def test_validation_error_logs_check_names_only(monkeypatch, caplog):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from genai import explain as module
    _mock_openai(monkeypatch)
    report = module.explain(_evidence()).replace("r1", "PRIVATE_REPORT_TEXT")

    assert not module._safe_report(report, _evidence())
    assert re.search(r"decision_reasons", caplog.text)
    assert "PRIVATE_REPORT_TEXT" not in caplog.text


@pytest.mark.parametrize(("qr_found", "phrase"), [
    (True, "QR code detected."),
    (False, "QR code was not detected."),
])
def test_qr_report_renders_detection_state(monkeypatch, qr_found, phrase):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from genai import explain as module
    evidence = _evidence()
    evidence["qr"] = {"qr_found": qr_found, "urls": []}
    _mock_openai(monkeypatch)

    report = module.explain(evidence)
    assert phrase in report
    assert "Missing QR codes or URLs are neutral." in report
    assert module._safe_report(report, evidence)


@pytest.mark.parametrize(("valid_syntax", "reachable", "syntax_text", "reachability_text"), [
    (True, True, "syntax valid: True", "reachability: reachable"),
    (True, False, "syntax valid: True", "reachability: unreachable"),
    (True, None, "syntax valid: True", "reachability: unknown"),
    (False, True, "syntax valid: False", "reachability: reachable"),
    (False, False, "syntax valid: False", "reachability: unreachable"),
    (False, None, "syntax valid: False", "reachability: unknown"),
])
def test_url_report_renders_syntax_and_each_reachability_state(
        monkeypatch, valid_syntax, reachable, syntax_text, reachability_text):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from genai import explain as module
    evidence = _evidence()
    evidence["qr"] = {"qr_found": True, "urls": [{
        "url": "https://verify.example/cert/123",
        "valid_syntax": valid_syntax, "reachable": reachable,
    }]}

    _mock_openai(monkeypatch)
    report = module.explain(evidence)
    assert "QR code detected" in report
    assert "https://verify.example/cert/123" in report
    assert syntax_text in report
    assert reachability_text in report
    assert "Missing QR codes or URLs are neutral." in report
    assert "areneutral" not in report
    assert module._safe_report(report, evidence)
