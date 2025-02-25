import streamlit as st
from Levenshtein import distance as levenshtein_distance
import re

# Function to clean and normalize words (lowercase, but keep spaces)
def clean_word(word):
    return re.sub(r'[^\w\s]', '', word).lower()  # Keep spaces, remove special characters

# Function to find closest matches and Levenshtein-1 distance words (Word-based Sliding Window)
def find_closest_matches(search_word, ocr_text):
    """Finds the closest word/phrase in OCR text with Levenshtein-1 distance using word-based sliding window."""
    
    search_clean = clean_word(search_word)
    words = ocr_text.split()  # Split OCR text into words

    best_match = None
    best_distance = float('inf')
    levenshtein_1_matches = set()

    # Sliding window over words (unigrams, bigrams, trigrams)
    for window_size in range(1, 4):  # Try 1-word, 2-word, and 3-word phrases
        for i in range(len(words) - window_size + 1):
            candidate = " ".join(words[i:i + window_size])  # Form phrases
            candidate_clean = clean_word(candidate)

            dist = levenshtein_distance(candidate_clean, search_clean)

            if dist == 1:
                levenshtein_1_matches.add(candidate.strip())  # Store cleaned match

            if dist < best_distance:
                best_match = candidate
                best_distance = dist

    return best_match, sorted(levenshtein_1_matches)  # Sorted for better readability

# Streamlit UI
st.title("🔍 OCR Text Correction and Analysis")

ocr_text = st.text_area("📄 Enter OCR Text:", height=200)
search_word = st.text_input("🔎 Enter Search Word:")

if st.button("🔍 Find Matches"):
    if not ocr_text or not search_word:
        st.warning("Please enter OCR text and a search word!")
    else:
        closest_match, lev_1_matches = find_closest_matches(search_word, ocr_text)

        # Display results
        st.write("## 📊 Results")
        st.write(f"**Search Word:** {search_word}")
        st.write(f"**Closest Match:** {closest_match if closest_match else 'N/A'}")
        st.write(f"**Levenshtein-1 Matches:** {', '.join(lev_1_matches) if lev_1_matches else 'None'}")
        