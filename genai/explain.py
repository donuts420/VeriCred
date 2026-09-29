"""Template-first explainer. Optional LLM rephrase only; never new evidence."""
import os


def _template(ev):
    det, qr, dec = ev["det"], ev["qr"], ev["decision"]
    fields = ev.get("fields", {}) or {}
    L = [f"## Verification result\n{dec['status']}", ""]
    L.append("### Certificate information")
    known = [(k.replace("_", " ").title(), v) for k, v in fields.items() if v]
    L += [f"- {k}: {v}" for k, v in known] or ["- No fields confidently extracted."]
    L += ["", "### OCR analysis",
          f"- {ev.get('ocr_chars', 0)} characters extracted.",
          "- Raw text available in the OCR section above.", ""]
    L.append("### QR / URL analysis")
    L.append(f"- QR code: {'found' if qr.get('qr_found') else 'not detected (neutral)'}.")
    urls = qr.get("urls", [])
    L += ["- URLs: none detected (neutral)."] if not urls else [
        f"- {u['url']} (syntax valid: {u['valid_syntax']}, reachable: {u['reachable']})."
        for u in urls]
    L += ["", "### Forgery analysis"]
    if not det.get("model_loaded"):
        L.append("- Detector unavailable — trained weights not loaded.")
    else:
        L.append(f"- {det['fake_count']} fake region(s), {det['true_count']} true region(s).")
        for b in det.get("boxes", []):
            x1, y1, x2, y2 = b["bbox"]
            L.append(f"- {b['label']} region, confidence {b['confidence']:.2f}, "
                     f"box [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}].")
    L += ["", "### Why this result?",
          " ".join(dec.get("reasons", [])), "",
          "### Limitations",
          "- The detector reports visual/document evidence, not proof of fraud or legitimacy.",
          "- Absence of fake regions is not proof the certificate is valid.",
          "- Missing QR/URL is neutral; many legitimate certificates lack them."]
    return "\n".join(L)


def explain(evidence):
    """Template always; LLM (if LLM_API_KEY set) may only rephrase this text."""
    base = _template(evidence)
    if not os.environ.get("LLM_API_KEY"):
        return base
    try:  # ponytail: best-effort rephrase; any failure returns the template
        from urllib.request import Request, urlopen
        import json
        req = Request("https://api.openai.com/v1/chat/completions",
                      data=json.dumps({
                          "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                          "messages": [
                              {"role": "system", "content": (
                                  "Rephrase the verification report in plain language. "
                                  "Do not add facts, numbers, or verdicts beyond the text given.")},
                              {"role": "user", "content": base}],
                      }).encode(),
                      headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}",
                               "Content-Type": "application/json"})
        reply = json.loads(urlopen(req, timeout=20).read())
        return reply["choices"][0]["message"]["content"]
    except Exception:
        return base
