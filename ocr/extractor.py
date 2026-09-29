from PIL import Image, ImageFilter
import pytesseract
import pdfplumber
import os
import re

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

# TO EXTRACT TEXT FROM PDF

def extract_text_from_pdf(file_path):
    try:
        text =""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
                else:
                    print("No text found in the PDF")
        return {
            "raw_text": text,
            "document_type": "pdf"
        }
    except Exception as e:
        print("Error: "+ str(e))


# TO EXTRACT TEXT FROM IMAGE

def ocr_image(file_path):
    try:
        img = Image.open(file_path)
        # To change to grayscale
        img = img.convert('L') 
        # To remove noise
        img = img.filter(ImageFilter.MedianFilter())
        img = img.point(lambda x: 0 if x < 140 else 255)
        text = pytesseract.image_to_string(img)
        return {
            "raw_text": text,
            "document_type": "image"
        }
    except Exception as e:
        print("Error: "+ str(e))


def _ocr_with_fallback(img):
    """Direct OCR first; aggressive preprocessing only as fallback."""
    text = pytesseract.image_to_string(img)
    if text and text.strip():
        return text
    gray = img.convert("L").filter(ImageFilter.MedianFilter())
    gray = gray.point(lambda x: 0 if x < 140 else 255)
    return pytesseract.image_to_string(gray)


# REUSABLE ENTRY POINT (used by the VeriCred pipeline)

def _ocr_scanned_pdf(file_path, dpi=200):
    """Render PDF pages to images and OCR them (for scanned/image-only PDFs)."""
    import fitz  # PyMuPDF
    doc = fitz.open(file_path)
    texts = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        texts.append(_ocr_with_fallback(img))
    return "\n".join(texts)


def extract_certificate(file_path):
    """Dispatch on extension; scanned PDFs fall back to render+OCR."""
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Invalid Document. Must be a PDF or Image")
    if extension == ".pdf":
        data = extract_text_from_pdf(file_path)
        text = (data.get("raw_text") or "") if data else ""
        if not text.strip():  # scanned / image-only PDF
            text = _ocr_scanned_pdf(file_path)
            return {"raw_text": text, "document_type": "pdf_scanned"}
        return data
    try:
        img = Image.open(file_path).convert("RGB")
        return {"raw_text": _ocr_with_fallback(img), "document_type": "image"}
    except Exception:
        return ocr_image(file_path)


def extract_fields(raw_text):
    """Conservative field guesses; missing fields are None (never invented)."""
    text = raw_text or ""
    fields = {
        "student_name": None,
        "degree": None,
        "institution": None,
        "year": None,
        "certificate_number": None,
    }
    m = re.search(r"(?:name of (?:the )?(?:student|candidate)|student name|awarded to|this is to certify that)\s*[:\-]?\s*([A-Z][A-Za-z .']{2,60})", text, re.I)
    if m:
        fields["student_name"] = m.group(1).strip()
    m = re.search(r"\b(Bachelor|Master|B\.?\s?Tech|M\.?\s?Tech|BSc|MSc|MBA|Ph\.?D|Diploma)[^,\n]{0,60}", text, re.I)
    if m:
        fields["degree"] = m.group(0).strip()
    m = re.search(r"\b([A-Z][A-Za-z .']*(?:University|Institute|College|Board)[A-Za-z .']*)", text, re.I)
    if m:
        fields["institution"] = re.sub(r"^(?:[a-z][\w.']*\s+)+", "", m.group(1)).strip()
    m = re.search(r"\b(19|20)\d{2}\b", text)
    if m:
        fields["year"] = m.group(0)
    m = re.search(r"(?:certificate|cert\.?|registration|roll|enrolment|enrollment)\s*(?:no|number|id)\s*[:\-]?\s*([A-Za-z0-9/\-]{3,30})", text, re.I)
    if m:
        fields["certificate_number"] = m.group(1).strip()
    return fields


# MAIN PROGRAM (CLI only; importing this module must not prompt)

if __name__ == "__main__":
    file_path = input("Enter the file path: ")
    try:
        data = extract_certificate(file_path)
        print(data.get("raw_text", ""))
    except Exception as e:
        print("Error: " + str(e))