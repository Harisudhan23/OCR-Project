import google.generativeai as genai
from PIL import Image
import os
import Levenshtein  # For calculating edit distance
import configparser

# Load configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Load API Key from the config file
GOOGLE_API_KEY = config['google_api']['api_key']
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in config file.")

genai.configure(api_key=GOOGLE_API_KEY)

def extract_text_from_handwritten_image(image_path):
    """Extracts text from a handwritten image using the Gemini Flash Vision model."""

    model = genai.GenerativeModel('gemini-2.0-flash')

    try:
        img = Image.open(image_path)  # Use passed image path
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    prompt = """
    You are an expert in deciphering handwritten text.
    Carefully examine the image and extract all the both handwritten text and digital text accurately.
    Preserve the original formatting as much as possible, including line breaks and spacing.
    Return ONLY the extracted text.  Do not add any additional text, explanations, or greetings.
    If the text seems to follow some sort of pattern, document or list or numbered steps, please attempt to preserve it, while extracting it.
    If you cannot understand a word, make your best educated guess. If that seems entirely impossible, insert a question mark.
    """

    try:
        response = model.generate_content([prompt, img], stream=False)
        response.resolve()
        return response.text
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return None


def calculate_cer(reference, hypothesis):
    """Calculates the Character Error Rate (CER)."""
    return Levenshtein.distance(reference, hypothesis) / len(reference)

def calculate_wer(reference, hypothesis):
    """Calculates the Word Error Rate (WER)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)


if __name__ == "__main__":
    # Read image path and ground truth text from config file
    image_path = config['image']['image_path']
    ground_truth_text_path = config['settings']['ground_truth_text']
    with open(ground_truth_text_path, 'r') as f:
      ground_truth_text = f.read()

    extracted_text = extract_text_from_handwritten_image(image_path)

    if extracted_text:
        print("Extracted Text (Gemini):\n", extracted_text)

        # Calculate CER and WER
        cer = calculate_cer(ground_truth_text, extracted_text)
        wer = calculate_wer(ground_truth_text, extracted_text)

        print(f"\nCharacter Error Rate (CER): {cer:.4f}")
        print(f"Word Error Rate (WER): {wer:.4f}")
    else:
        print("Text extraction failed.")
