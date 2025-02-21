import streamlit as st
import re
import spacy
import nltk
from nltk.corpus import words
from difflib import get_close_matches
from Levenshtein import distance as lev

# Load NLP model and English words
nlp = spacy.load("en_core_web_sm")
nltk.download("words")
word_list = set(words.words())

# OCR Correction Dictionary (Common OCR mistakes)
OCR_MAP = str.maketrans("105", "ios")  # Example: "Art1ficial" → "Artificial"

def correct_ocr(word):
    """Fix common OCR errors like '1' → 'i', '0' → 'o'."""
    return word.translate(OCR_MAP) # Normalize to lowercase

def extract_words(text):
    """Extract words from text and apply OCR correction."""
    doc = nlp(text)
    words_in_text = set(token.text.lower() for token in doc if token.is_alpha or token.text.isalnum())

    # Apply OCR correction and store both versions
    corrected_words = {correct_ocr(word) for word in words_in_text}
    
    return words_in_text | corrected_words  # Merge original and corrected words

def find_keyword_matches(text, keyword, max_distance=1):
    """Find exact matches, OCR variants, and words with minor errors."""
    
    words_in_text = extract_words(text)

    # Convert keyword to lowercase and apply OCR correction
    corrected_keyword = correct_ocr(keyword)

    # Exact match
    keyword_in_text = {word for word in words_in_text if word == corrected_keyword}

    # Find closest matches with a stricter cutoff
    closest_matches = set(get_close_matches(corrected_keyword, words_in_text, cutoff=0.7))

    # Find words with Levenshtein distance ≤ max_distance
    levenshtein_matches = {word for word in words_in_text if lev(corrected_keyword, word) <= max_distance}

    # Find words with minor truncations or extensions
    prefix_suffix_matches = {word for word in words_in_text if word.startswith(corrected_keyword) or corrected_keyword.startswith(word)}

    # Combine all results and remove duplicates
    search_terms = keyword_in_text | closest_matches | levenshtein_matches | prefix_suffix_matches

    # Ensure no false positives
    search_terms = {word for word in search_terms if lev(corrected_keyword, word) <= max_distance}
    print("🔍 Extracted Words from Text:", words_in_text)

    matches = []
    for match in search_terms:
        for m in re.finditer(re.escape(match), text, re.IGNORECASE):  
            matches.append(match)

    return sorted(set(matches))

st.title("🔍 Keyword Finder with OCR & Levenshtein Matching")

text_input = st.text_area("📜 Enter the text:", height=300)
keyword_input = st.text_input("🔑 Enter the keyword to search:")

if st.button("Find Matches"):
    if text_input and keyword_input:
        matches = find_keyword_matches(text_input, keyword_input)
        st.subheader("🔎 Words Considered for Search")
        st.write(", ".join(matches) if matches else "No matches found.")
    else:
        st.warning("⚠️ Please enter both text and keyword!")
