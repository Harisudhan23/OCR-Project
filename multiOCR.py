import streamlit as st
from Levenshtein import distance as levenshtein_distance
import re
import multiprocessing

def clean_word(word, preserve_spaces=False, lowercase_only=False):
    """
    Cleans input text:
    - Removes special characters (including colons `:`, commas `,`, etc.)
    - Preserves spaces if `preserve_spaces=True`
    - Converts to lowercase only if `lowercase_only=True`
    """
    word = re.sub(r'^\d+\.\s*', '', word)  # Remove leading numbers & periods
    word = re.sub(r'\s+', ' ', word).strip()  # Normalize spaces

    if lowercase_only:
        return re.sub(r'[^\w\s]', '', word.lower()) if preserve_spaces else re.sub(r'[^\w]', '', word.lower())
    return word  # Return original case if lowercase_only=False

def sanitize_match(word):
    """
    Cleans the matched word by removing trailing special characters but keeps the original case.
    """
    return re.sub(r'[:;,!?]+$', '', word).strip()  # Remove trailing punctuation

def process_search_word(search_word, words):
    """
    Finds the closest match for a single search word using multiprocessing.
    """
    search_clean = clean_word(search_word, preserve_spaces=True, lowercase_only=True)
    best_match = None
    best_distance = float('inf')
    lev_1_matches = set()
    lev_2_matches = set()

    # Sliding window over words (unigrams, bigrams, trigrams)
    for window_size in range(1, 4):  # Try 1-word, 2-word, and 3-word phrases
        for i in range(len(words) - window_size + 1):
            candidate = " ".join(words[i:i + window_size])  # Form phrases
            candidate_clean = clean_word(candidate, preserve_spaces=True, lowercase_only=True)

            dist = levenshtein_distance(candidate_clean, search_clean)

            if dist == 1:
                lev_1_matches.add(sanitize_match(candidate))  # Sanitize before adding
            elif dist == 2:
                lev_2_matches.add(sanitize_match(candidate))  # Sanitize before adding

            if dist < best_distance:
                best_match = sanitize_match(candidate)  # Sanitize before assigning
                best_distance = dist

    return search_word, {
        "closest_match": best_match if best_match else 'N/A',
        "levenshtein_1_matches": sorted(lev_1_matches),
        "levenshtein_2_matches": sorted(lev_2_matches)
    }

def find_closest_matches(search_words, ocr_text):
    """
    Uses multiprocessing to find the closest word/phrase in OCR text for multiple search words.
    """
    words = ocr_text.split()  # Split OCR text into words
    results = {}

    # Use multiprocessing to process each search word in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        matches = pool.starmap(process_search_word, [(word, words) for word in search_words])

    # Convert list to dictionary
    results = dict(matches)
    return results

st.title("🔍 OCR Text Correction and Analysis (Multiprocessing)")

ocr_text = st.text_area("📄 Enter OCR Text:", height=200)
search_words_input = st.text_input("🔎 Enter Search Words (comma-separated):")

if st.button("🔍 Find Matches"):
    if not ocr_text or not search_words_input:
        st.warning("Please enter OCR text and search words!")
    else:
        search_words = [word.strip() for word in search_words_input.split(",")]  # Convert input into a list of words
        
        # Run multiprocessing-based OCR matching
        results = find_closest_matches(search_words, ocr_text)

        st.write("## 📊 Results")
        for word, result in results.items():
            st.write(f"### 🔹 Search Word: **{word}**")
            st.write(f"✅ **Closest Match:** {result['closest_match']}")
            st.write(f"✅ **Levenshtein-1 Matches:** {', '.join(result['levenshtein_1_matches']) if result['levenshtein_1_matches'] else 'None'}")
            st.write(f"✅ **Levenshtein-2 Matches:** {', '.join(result['levenshtein_2_matches']) if result['levenshtein_2_matches'] else 'None'}")
