import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import easyocr
import cv2
from gtts import gTTS
import base64

st.set_page_config(page_title="AI Vision Mate OCR", layout="wide")
st.title("AI Vision Mate: OCR for Urdu + Sindhi + English")
st.write("Capture text from your camera. Accuracy optimized with preprocessing.")

# EasyOCR reader
reader = easyocr.Reader(['en', 'ur', 'ar'], gpu=False)

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.resize((img.width*2, img.height*2), Image.LANCZOS)
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    img = img.filter(ImageFilter.SHARPEN)
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

def extract_text(img: Image.Image) -> str:
    img = preprocess_image(img)
    try:
        result = reader.readtext(np.array(img), detail=0, paragraph=True)
        text = "\n".join(result).strip()
        if text:
            return text
    except:
        return "No text detected"
    return "No text detected"

def speak_text(text: str):
    tts = gTTS(text=text, lang='en')
    tts.save("tts.mp3")
    audio_file = open("tts.mp3", "rb").read()
    b64 = base64.b64encode(audio_file).decode()
    st.audio(f"data:audio/mp3;base64,{b64}", format="audio/mp3")

camera_image = st.camera_input("Capture Text Image")

if camera_image is not None:
    image = Image.open(camera_image)
    st.image(image, caption="Captured Image", use_column_width=True)
    
    with st.spinner("Extracting text..."):
        text = extract_text(image)
    
    st.text_area("Extracted Text:", text, height=200)

    if st.button("Read Aloud"):
        speak_text(text)
