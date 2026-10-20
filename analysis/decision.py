"""Evidence combiner. decide(det, qr_url, conf) — hash is NOT an input."""

TAMPERED = "Potentially Tampered"
CLEAR = "No high-confidence tampering detected"
INCONCLUSIVE = "Inconclusive / analysis unavailable"


def decide(det, qr_url, conf=0.6):
    reasons = []
    if not det.get("model_loaded"):
        reasons.append("Forgery detector unavailable (no trained weights loaded).")
        return {"status": INCONCLUSIVE, "reasons": reasons}
    fc, mc = det.get("fake_count", 0), det.get("max_fake_conf", 0.0)
    if fc >= 1 and mc >= conf:
        reasons.append(
            f"{fc} fake region(s) detected; strongest confidence "
            f"{mc:.2f} >= threshold {conf:.2f}.")
        reasons.append("This is visual evidence of possible tampering, not proof of fraud.")
        return {"status": TAMPERED, "reasons": reasons}
    if fc >= 1:
        reasons.append(
            f"{fc} weak fake signal(s) below threshold "
            f"(max {mc:.2f} < {conf:.2f}); not counted as tampering.")
    else:
        reasons.append("No fake regions detected at the current threshold.")
    reasons.append("Absence of detections is not proof the certificate is authentic.")
    if not (qr_url or {}).get("qr_found") and not (qr_url or {}).get("urls"):
        reasons.append("No QR code or URL found; treated as neutral (many genuine certificates lack them).")
    return {"status": CLEAR, "reasons": reasons}
