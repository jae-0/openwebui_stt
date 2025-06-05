from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF

# 일반 이미지(JPG, PNG 등) → 텍스트
def extract_text_from_image(image_path: str) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang="eng+kor").strip()

# PDF(텍스트 기반) → 텍스트
def extract_text_from_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text.strip()

# PDF(스캔 이미지 기반) → OCR
def extract_text_from_pdf_image(pdf_path: str) -> str:
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang="eng+kor") + "\n"
    return text.strip()
