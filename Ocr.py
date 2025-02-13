import os
import cv2
import numpy as np
import configparser
from dotenv import load_dotenv
from PIL import Image
import Levenshtein
from paddleocr import PaddleOCR

load_dotenv()

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def preprocess_image(image_path):
    """Preprocesses the image for better OCR."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Noise Removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Deskewing using hough lines
    coords = np.column_stack(np.where(opening > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed

def extract_text_from_handwritten_image(image_path):
    """Extracts text from a handwritten image using PaddleOCR."""
    try:
        preprocessed_image = preprocess_image(image_path)
        pil_img = Image.fromarray(preprocessed_image)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    
    # Convert image to OCR-compatible format
    extracted_text_results = ocr.ocr(np.array(pil_img), cls=True)
    extracted_text = " ".join([word_info[1][0] for line in extracted_text_results for word_info in line])
    return extracted_text.strip()

def calculate_cer(reference, hypothesis):
    """Calculates the Character Error Rate (CER)."""
    return Levenshtein.distance(reference, hypothesis) / len(reference) if reference else 1.0

def calculate_wer(reference, hypothesis):
    """Calculates the Word Error Rate (WER)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words) if ref_words else 1.0

if __name__ == "__main__":
    # Read image path and ground truth text from config file
    image_path = config['image']['image_path']
    ground_truth_text_path = config['settings']['ground_truth_text']
    
    try:
        with open(ground_truth_text_path, 'r', encoding='utf-8') as f:
            ground_truth_text = f.read()
    except FileNotFoundError:
        print(f"Error: Ground truth text file not found at {ground_truth_text_path}")
        ground_truth_text = ""
    except Exception as e:
        print(f"Error reading ground truth text file: {e}")
        ground_truth_text = ""
    
    extracted_text = extract_text_from_handwritten_image(image_path)
    
    if extracted_text:
        print("Extracted Text:\n", extracted_text)
        
        # Calculate CER and WER
        cer = calculate_cer(ground_truth_text, extracted_text)
        wer = calculate_wer(ground_truth_text, extracted_text)
        
        print(f"\nCharacter Error Rate (CER): {cer:.4f}")
        print(f"\nWord Error Rate (WER): {wer:.4f}")
    else:
        print("Text extraction failed.")
