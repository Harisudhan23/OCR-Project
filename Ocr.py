import cv2
import pytesseract
import numpy as np
from rapidfuzz import fuzz
import jiwer

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  

def preprocess_image(image_path):
    """
    Preprocesses the image to improve OCR accuracy.
    - Converts to grayscale
    - Applies adaptive thresholding
    - Removes noise
    """
    img = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    processed_path = "processed_image.jpg"
    cv2.imwrite(processed_path, thresh)  
    return processed_path

def extract_text(image_path):
    """
    Extracts text from an image using Tesseract OCR.
    """
    text = pytesseract.image_to_string(image_path)
    return text.strip()


if __name__ == "__main__":
    image_path = "C:/Users/Harihara Sudhan N/Downloads/sample_image.jpg"  
    processed_image = preprocess_image(image_path)  
    extracted_text = extract_text(processed_image)  

    print("Extracted Text:\n", extracted_text)
