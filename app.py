import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import easyocr
import pytesseract
import cv2
import pyttsx3
import os

# -----------------------------
# OCR Setup
# -----------------------------
st.set_page_config(page_title="AI Vision Mate OCR", layout="wide")
st.title("AI Vision Mate: OCR for Urdu + Sindhi + English")

st.write("Capture text from your camera. Accuracy optimized with preprocessing.")

# EasyOCR reader (en=English, ur=Urdu, ar=Arabic for Sindhi)
reader = easyocr.Reader(['en', 'ur', 'ar'], gpu=False)

# Tesseract fallback
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    # Resize for better OCR
    img = img.resize((img.width*2, img.height*2), Image.LANCZOS)
    # Convert to grayscale
    img = img.convert('L')
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    # Sharpen image
    img = img.filter(ImageFilter.SHARPEN)
    # Deskew using OpenCV
    cv_img = np.array(img)
    _, thresh = cv2.threshold(cv_img, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size != 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        h, w = cv_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cv_img = cv2.warpAffine(cv_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        img = Image.fromarray(cv_img)
    return img

# -----------------------------
# OCR extraction function
# -----------------------------
def extract_text(img: Image.Image) -> str:
    img = preprocess_image(img)
    # EasyOCR
    try:
        text = reader.readtext(np.array(img), detail=0, paragraph=True)
        text = "\n".join(text).strip()
        if text and len(text) > 2:
            return text
    except:
        pass
    # Tesseract fallback
    try:
        text = pytesseract.image_to_string(img, lang='eng+urd').strip()
        if text:
            return text
    except:
        pass
    return "No text detected"

# -----------------------------
# Streamlit UI
# -----------------------------
camera_image = st.camera_input("Capture Text Image")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)
    
    with st.spinner("Extracting text..."):
        text = extract_text(image)
    
    st.text_area("Extracted Text:", text, height=200)

    # Optional: Text-to-Speech
    if st.button("Read Aloud"):
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
