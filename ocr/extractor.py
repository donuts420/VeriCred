from PIL import Image, ImageFilter
import pytesseract
import pdfplumber
import os

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


# MAIN PROGRAM

file_path = input("Enter the file path: ")
try:
    extension = os.path.splitext(file_path)[1].lower()
    if extension == ".pdf":
        print("PDF Detected....\n")
        data = extract_text_from_pdf(file_path)
    elif extension in [".png",".jpg",".jpeg"]:
        print("Image Detected....\n")
        data = ocr_image(file_path)
    else:
        raise ValueError(
            "Invalid Document. Must be a PDF or Image"
        )
except Exception as e:
    print("Error: "+ str(e))