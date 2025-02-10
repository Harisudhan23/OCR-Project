import streamlit as st
import cv2
import pytesseract
import Levenshtein
import numpy as np
from PIL import Image

# Set Tesseract Path (Update if necessary)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image):
    """Convert image to grayscale and apply thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(image):
    """Perform OCR on the image using Tesseract."""
    processed_img = preprocess_image(image)
    text = pytesseract.image_to_string(processed_img, config='--psm 6')
    return text.strip()

def calculate_cer(ocr_text, reference_text):
    """Calculate Character Error Rate (CER)."""
    ocr_text = ocr_text.replace(" ", "")
    reference_text = reference_text.replace(" ", "")
    return Levenshtein.distance(ocr_text, reference_text) / max(len(reference_text), 1)

def calculate_wer(ocr_text, reference_text):
    """Calculate Word Error Rate (WER)."""
    ocr_words = ocr_text.split()
    ref_words = reference_text.split()
    return Levenshtein.distance(" ".join(ocr_words), " ".join(ref_words)) / max(len(ref_words), 1)

# Streamlit UI
def main():
   st.title("OCR & Error Rate Calculator")

# Upload Image
   uploaded_file = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])

   if uploaded_file:
    # Read image
      image = Image.open(uploaded_file)
      image_np = np.array(image)
    
      st.image(image, caption="Uploaded Image",  use_container_width=True)
    
      # Extract text
      ocr_text = extract_text(image_np)
    
    # Display extracted text
      st.subheader("📝 Extracted Text")
      st.text_area("OCR Output", ocr_text, height=200)
    
    # Input ground truth text
      reference_text = st.text_area("✍️ Enter Ground Truth (Reference Text)", height=200)
    
      if st.button("Calculate Error Rates"):
          if reference_text:
              cer = calculate_cer(ocr_text, reference_text)
              wer = calculate_wer(ocr_text, reference_text)
            
              st.subheader("📊 Error Rate Results")
              st.write(f"**Character Error Rate (CER):** {cer:.4f}")
              st.write(f"**Word Error Rate (WER):** {wer:.4f}")
          else:
              st.error("Please enter the reference text to calculate CER & WER.")

if __name__ == "__main__":
    main()        
