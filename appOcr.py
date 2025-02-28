import streamlit as st
from Levenshtein import distance as levenshtein_distance
import re

def clean_word(word):
    """
    Cleans input text by:
    - Removing special characters (except spaces)
    - Converting to lowercase
    - Removing leading numbers and periods (e.g., "1. Machine Learning" → "Machine Learning")
    """
    word = re.sub(r'^\d+\.\s*', '', word)  # Remove leading numbers & periods
    return re.sub(r'[^\w\s]', '', word).lower()  # Remove special characters, keep spaces

def find_closest_matches(search_words, ocr_text):
    """
    Finds the closest word/phrase in OCR text for multiple search words using a word-based sliding window.
    Includes Levenshtein-1 and Levenshtein-2 distance matches.
    """
    words = ocr_text.split()  # Split OCR text into words
    results = {}

    for search_word in search_words:
        search_clean = clean_word(search_word)
        best_match = None
        best_distance = float('inf')
        lev_1_matches = set()
        lev_2_matches = set()

        # Sliding window over words (unigrams, bigrams, trigrams)
        for window_size in range(1, 4):  # Try 1-word, 2-word, and 3-word phrases
            for i in range(len(words) - window_size + 1):
                candidate = " ".join(words[i:i + window_size])  # Form phrases
                candidate_clean = clean_word(candidate)

                dist = levenshtein_distance(candidate_clean, search_clean)

                if dist == 1:
                    lev_1_matches.add(candidate.strip())  # Store cleaned match
                elif dist == 2:
                    lev_2_matches.add(candidate.strip())  # Store cleaned match

                if dist < best_distance:
                    best_match = candidate
                    best_distance = dist

        results[search_word] = {
            "closest_match": best_match if best_match else 'N/A',
            "levenshtein_1_matches": sorted(lev_1_matches),
            "levenshtein_2_matches": sorted(lev_2_matches)  # Sort for better readability
        }

    return results  # Return results for all search words

st.title("🔍 OCR Text Correction and Analysis")

ocr_text = st.text_area("📄 Enter OCR Text:", height=200)
search_words_input = st.text_input("🔎 Enter Search Words (comma-separated):")

if st.button("🔍 Find Matches"):
    if not ocr_text or not search_words_input:
        st.warning("Please enter OCR text and search words!")
    else:
        search_words = [word.strip() for word in search_words_input.split(",")]  # Convert input into a list of words
        results = find_closest_matches(search_words, ocr_text)

        st.write("## 📊 Results")
        for word, result in results.items():
            st.write(f"### 🔹 Search Word: **{word}**")
            st.write(f"**Closest Match:** {result['closest_match']}")
            st.write(f"**Levenshtein-1 Matches:** {', '.join(result['levenshtein_1_matches']) if result['levenshtein_1_matches'] else 'None'}")
            st.write(f"**Levenshtein-2 Matches:** {', '.join(result['levenshtein_2_matches']) if result['levenshtein_2_matches'] else 'None'}")
