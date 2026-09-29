"""VeriCred: academic certificate verification prototype."""
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from ocr.extractor import extract_certificate, extract_fields
from detection.predict import predict
from qr_validation import analyze as qr_analyze
from analysis.decision import decide
from genai.explain import explain

FIELD_LABELS = {"student_name": "Student Name", "degree": "Degree",
                "institution": "Institution", "year": "Year",
                "certificate_number": "Certificate Number"}


def verify_file(data: bytes, suffix: str, conf: float):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        path = f.name
    ocr = extract_certificate(path) or {"raw_text": "", "document_type": "?"}
    text = ocr.get("raw_text") or ""
    fields = extract_fields(text)
    image = Image.open(path).convert("RGB") if suffix != ".pdf" else None
    if image is None:  # PDFs: render page 1 for visual/QR analysis
        import fitz
        doc = fitz.open(path)
        pix = doc[0].get_pixmap(dpi=150)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    det = predict(image, conf=conf)
    qr = qr_analyze(image, text)
    decision = decide(det, qr, conf)
    report = explain({"fields": fields, "ocr_chars": len(text),
                      "det": det, "qr": qr, "decision": decision})
    return {"ocr": ocr, "fields": fields, "image": image, "det": det,
            "qr": qr, "decision": decision, "report": report}


def main():
    st.title("VeriCred — Academic Certificate Verification")
    st.caption("Prototype: OCR + localized forgery regions + QR/URL evidence. "
               "Nothing here proves authenticity.")
    up = st.file_uploader("Upload certificate", type=["pdf", "png", "jpg", "jpeg"])
    conf = st.slider("Fake-region confidence threshold (tunable, not validated)",
                     0.05, 0.95, 0.50, 0.05)
    if up and st.button("Verify certificate"):
        res = verify_file(up.getvalue(), Path(up.name).suffix.lower(), conf)
        st.subheader("Certificate preview")
        st.image(res["image"], use_container_width=True)

        st.subheader("Certificate Information")
        for key, label in FIELD_LABELS.items():
            st.write(f"- **{label}:** {res['fields'].get(key) or 'Not detected'}")

        with st.expander("OCR text"):
            st.write(f"{len(res['ocr'].get('raw_text') or '')} characters extracted.")
            st.text(res["ocr"].get("raw_text") or "(no text extracted)")

        st.subheader("QR / URL")
        st.write(f"- **QR code:** {'Detected' if res['qr']['qr_found'] else 'Not detected'}")
        urls = res["qr"]["urls"]
        if not urls:
            st.write("- **URLs:** None detected")
        for u in urls:
            st.write(f"- **URL:** {u['url']} (syntax valid: {u['valid_syntax']}, "
                     f"reachable: {u['reachable']})")
        st.caption("Missing QR/URL is neutral — not evidence of forgery.")

        st.subheader("Forgery detection")
        if not res["det"]["model_loaded"]:
            st.warning("Forgery detector unavailable — trained weights not loaded.")
        else:
            st.image(res["det"]["annotated"],
                     caption="Detected regions (red=fake, green=true)",
                     use_container_width=True)
            st.write(f"Fake regions: **{res['det']['fake_count']}** | "
                     f"True regions: **{res['det']['true_count']}**")
            if res["det"]["boxes"]:
                st.table([{"label": b["label"],
                           "confidence": round(b["confidence"], 3),
                           "bbox": [round(v, 1) for v in b["bbox"]]}
                          for b in res["det"]["boxes"]])

        st.subheader("Final status")
        status = res["decision"]["status"]
        if status == "Potentially Tampered":
            st.error(status)
            st.write("Potential tampering detected because the forgery detector "
                     "identified one or more high-confidence regions labeled as fake.")
        elif status.startswith("Inconclusive"):
            st.warning(status)
            st.write("Inconclusive because the forgery detector has not been "
                     "trained/loaded. OCR and QR/URL analysis may still be available, "
                     "but visual forgery analysis could not be performed.")
        else:
            st.success(status)
            st.write("No high-confidence tampering was detected by the visual detector. "
                     "This does not prove that the certificate is authentic.")
        for r in res["decision"]["reasons"]:
            st.write("- " + r)

        st.subheader("Explanation")
        st.markdown(res["report"])


if __name__ == "__main__":
    main()
