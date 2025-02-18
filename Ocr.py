import requests
import json
import base64
import configparser
import google.generativeai as genai  
import cv2
import numpy as np
import Levenshtein
import streamlit as st

# Load configuration from config.ini
def load_config(config_path="config.ini"):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# Preprocess the image before encoding (Denoising + Thresholding)
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.adaptiveThreshold(image, 150, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        image = cv2.resize(image, (800, 800))
        denoised = cv2.fastNlMeansDenoising(image, None, 30, 7)

        processed_path = "processed_image.png"
        cv2.imwrite(processed_path, denoised)
        return processed_path

    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return image_path  

# Encode image as Base64 (for local image files)
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {e}")
        return None

# Google Cloud Vision OCR function
def extract_text_google_vision(image_path, api_key, is_url=False):
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"

    if not is_url:
        image_path = preprocess_image(image_path)

    image_data = {"source": {"imageUri": image_path}} if is_url else {"content": encode_image(image_path)}

    if not image_data.get("content") and not is_url:
        print("Error: Could not encode image for OCR.")
        return ""

    request_data = {
        "requests": [
            {
                "image": image_data,
                "features": [{"type": "TEXT_DETECTION"}]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(request_data))

    if response.status_code == 200:
        response_data = response.json()
        extracted_text = response_data["responses"][0].get("fullTextAnnotation", {}).get("text", "").strip()
        print(f"\n[DEBUG] Raw Extracted Text:\n{extracted_text}")  # Debugging output
        return extracted_text
    else:
        print(f"Error {response.status_code}: {response.text}")
        return ""

# Medical-specific spelling correction using Gemini AI
def correct_with_llm(extracted_text, genai_api_key):
    # Configure Gemini API Key
    genai.configure(api_key=genai_api_key)

    prompt_template = """
    "You are an expert in correcting OCR errors across various domains, including medical, legal, financial, technical, and other specialized documents. The text provided below has been extracted from a scanned or handwritten document using OCR technology, but due to recognition errors, some words may be misspelled, misformatted, or inaccurately interpreted.

Your Task:
Correct Recognition Errors: Fix spelling mistakes, typographical errors, and misrecognized words while ensuring the original meaning remains intact.
Preserve Domain-Specific Terminology: Do not modify technical, medical, legal, or financial terms unless they contain clear OCR errors.
Enhance Readability: Ensure proper punctuation, grammar, and formatting while keeping the content faithful to the original.
Retain Structured Data: Maintain the exact structure of numerical values, dates, monetary amounts, units of measurement, and special characters *(e.g., "5mg", "₹10,000", "17/Oct/2022")*.
Avoid Unverified Changes: Do not add, remove, or guess missing words—only correct what is evidently incorrect in the OCR output.

    Extracted Text:
    "{text}"

    Corrected Medical Text:
    """

    formatted_prompt = prompt_template.format(text=extracted_text)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")  
        response = model.generate_content(formatted_prompt)

        corrected_text = response.text.strip()
        print(f"\n[DEBUG] Corrected Text:\n{corrected_text}")  # Debugging output
        return corrected_text

    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return extracted_text  

# Find closest matching words
def find_closest_words(extracted_text, genai_api_key, search_phrases,):
    # Configure Gemini API Key
    genai.configure(api_key=genai_api_key)

    # Prepare the search phrases as a list
    search_phrases = search_phrases.split(", ")  # Allow multiple search terms
    closest_matches = {}

    for phrase in search_phrases:
        # Define the prompt to pass to Gemini AI
        prompt = f"""
        You are an expert in understanding natural language and OCR text extraction. Given the extracted text below, 
        find the exact or closest matching phrase for the search phrase. The search phrase might have OCR errors or variations, 
        but you should prioritize finding the closest full match.

        Extracted Text:
        "{extracted_text}"

        Search Phrase: "{phrase}"

        Closest Match:
        """

        try:
            # Generate content using the Gemini AI model (gemini-2.0-flash)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)

            # Retrieve the closest match for the search phrase
            closest_match = response.text.strip()
            closest_matches[phrase] = closest_match

        except Exception as e:
            print(f"Error with Gemini API: {e}")
            closest_matches[phrase] = "Error finding match"

    return closest_matches

def calculate_cer(ground_truth, predicted_text):
    if not ground_truth:  # If ground truth is empty
        return 1 if predicted_text else 0
    return Levenshtein.distance(ground_truth, predicted_text) / len(ground_truth)

def calculate_wer(ground_truth, predicted_text):
    ground_truth_words = ground_truth.split()
    predicted_words = predicted_text.split()

    if not ground_truth_words:  # If ground truth is empty
        return 1 if predicted_words else 0

    return Levenshtein.distance((ground_truth_words), (predicted_words)) / len(ground_truth_words)

# Summarization using Gemini AI
def summarize_text(corrected_text, genai_api_key):
    genai.configure(api_key=genai_api_key)
    prompt = f"""
    "You are an expert in summarizing documents from various sectors, including medical, legal, financial, technical, and more. 
    Given the following corrected text, provide a concise summary that retains key information without altering the meaning.
    Your summary should be precise, clear, and appropriate for the relevant sector.

    Corrected Text:
    "{corrected_text}"

    Summary:
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")  # Use a lightweight model for speed
        response = model.generate_content(prompt)

        summary = response.text.strip()
        print(f"\n[DEBUG] Summary:\n{summary}")  # Debugging output
        return summary

    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return "Summary could not be generated."


# Main execution
def main():
    # Set the title for the page
    st.title("OCR with AI-powered Correction & Summarization")

    # Apply background color using custom CSS
    st.markdown("""
        <style>
            body {
                background-color: #f0f8ff;  /* Light blue background */
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border-radius: 12px;
                height: 40px;
            }
            .stTextInput input {
                background-color: #ffffff;
            }
            .stTextArea textarea {
                background-color: #f7f7f7;
            }
        </style>
        """, unsafe_allow_html=True)
    # Load API keys from config file
    config = load_config()
    google_api_key = config.get("GoogleCloud", "api_key", fallback="")
    genai_api_key = config.get("google_api", "api_key", fallback="")

    # Image upload section
    st.subheader("1. Upload Image for OCR")
    uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file:
        # Display image preview
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Search word input section
    st.subheader("2. Search for Keywords or Phrases")
    search_word = st.text_input("Enter the keyword or phrase to search:")

    # Ground truth input for evaluation
    st.subheader("3. Ground Truth for Error Evaluation (Optional)")
    ground_truth = st.text_area("Enter ground truth text (optional for evaluation)")

    # Process image and extract text when uploaded
    if uploaded_file:
        # Save uploaded file
        temp_file_path = "uploaded_image.png"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text using Google Vision API
        st.markdown("### Extracted Text:")
        extracted_text = extract_text_google_vision(temp_file_path, google_api_key)
        st.text_area("Extracted Text", extracted_text, height=150)

        if extracted_text:
            # Correct text with Gemini AI
            st.markdown("### Corrected Text (via AI-powered correction):")
            corrected_text = correct_with_llm(extracted_text, genai_api_key)
            st.text_area("Corrected Text", corrected_text, height=150)

            # Search for closest matches if search_word is provided
            if search_word:
                st.markdown("### Closest Matches:")
                closest_matches = find_closest_words(corrected_text, genai_api_key, search_word)
                for term, match in closest_matches.items():
                    st.write(f"**Closest match to '{term}':** {match}")

            # Summarize the corrected text
            st.markdown("### Summary of Corrected Text:")
            summary = summarize_text(corrected_text, genai_api_key)
            st.text_area("Summary", summary, height=100)

            # Evaluate using CER and WER if ground truth is provided
            if ground_truth:
                cer = calculate_cer(ground_truth, corrected_text)
                wer = calculate_wer(ground_truth, corrected_text)
                st.write(f"**Character Error Rate (CER):** {cer:.2f}")
                st.write(f"**Word Error Rate (WER):** {wer:.2f}")

if __name__ == "__main__":
    main()
