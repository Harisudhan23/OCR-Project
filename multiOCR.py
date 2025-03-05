import streamlit as st
import os
import re
import time
import multiprocessing
from datetime import datetime
from Levenshtein import distance as levenshtein_distance

class OCRTextCorrection:
    def __init__(self):
        self.ocr_text = ""
        self.search_words = []

    @staticmethod
    def clean_word(word, preserve_spaces=False, lowercase_only=False):
        word = re.sub(r'^\d+\.\s*', '', word)  # Remove leading numbers & periods
        word = re.sub(r'\s+', ' ', word).strip()  # Normalize spaces
        if lowercase_only:
            return re.sub(r'[^\w\s]', '', word.lower()) if preserve_spaces else re.sub(r'[^\w]', '', word.lower())
        return word

    @staticmethod
    def sanitize_match(word):
        return re.sub(r'[:;,!?]+$', '', word).strip()  # Remove trailing punctuation

    @staticmethod
    def process_search_word(args):
        core_id, search_word, words = args
        start_time = datetime.now()  # Start timestamp

        search_clean = OCRTextCorrection.clean_word(search_word, preserve_spaces=True, lowercase_only=True)
        best_match = None
        best_distance = float('inf')
        lev_1_matches = set()
        lev_2_matches = set()

        for window_size in range(1, 4):  # Unigrams, bigrams, trigrams
            for i in range(len(words) - window_size + 1):
                candidate = " ".join(words[i:i + window_size])
                candidate_clean = OCRTextCorrection.clean_word(candidate, preserve_spaces=True, lowercase_only=True)

                dist = levenshtein_distance(candidate_clean, search_clean)
                if dist == 1:
                    lev_1_matches.add(OCRTextCorrection.sanitize_match(candidate))
                elif dist == 2:
                    lev_2_matches.add(OCRTextCorrection.sanitize_match(candidate))
                if dist < best_distance:
                    best_match = OCRTextCorrection.sanitize_match(candidate)
                    best_distance = dist

        end_time = datetime.now()  # End timestamp
        execution_time = (end_time - start_time).total_seconds()
        
        return search_word, {
            "closest_match": best_match if best_match else 'N/A',
            "levenshtein_1_matches": sorted(lev_1_matches),
            "levenshtein_2_matches": sorted(lev_2_matches),
            "start_time": start_time.strftime('%H:%M:%S.%f'),
            "end_time": end_time.strftime('%H:%M:%S.%f'),
            "execution_time": execution_time
        }

    def find_closest_matches(self):
        words = self.ocr_text.split()
        num_cores = min(len(self.search_words), os.cpu_count())  # Limit to available cores
        pre_multiprocessing_time = datetime.now()  # ✅ Time before launching processes

        with multiprocessing.Pool(processes=num_cores) as pool:
            results_list = pool.map(OCRTextCorrection.process_search_word, [(i, word, words) for i, word in enumerate(self.search_words)])
        
        post_multiprocessing_time = datetime.now()  # ✅ Time after multiprocessing completes
        results = dict(results_list)
        total_time = (post_multiprocessing_time - pre_multiprocessing_time).total_seconds()
        
        return (
            results, 
            pre_multiprocessing_time.strftime('%H:%M:%S.%f'), 
            post_multiprocessing_time.strftime('%H:%M:%S.%f'), 
            total_time
        )

    def run(self):
        st.title("🔍 OCR Text Correction and Analysis (Optimized Multiprocessing)")

        self.ocr_text = st.text_area("📄 Enter OCR Text (2000+ words recommended):", height=300)
        search_words_input = st.text_input("🔎 Enter Search Words (comma-separated):")

        if st.button("🔍 Find Matches"):
            if not self.ocr_text or not search_words_input:
                st.warning("Please enter OCR text and search words! ⚠️")
            else:
                self.search_words = [word.strip() for word in search_words_input.split(",")]

                # Run multiprocessing-based OCR matching
                results, pre_time, post_time, total_time = self.find_closest_matches()

                st.write("## 📊 Results")
                for word, result in results.items():
                    st.write(f"### 🔹 Search Word: **{word}**")
                    st.write(f"✅ **Closest Match:** {result['closest_match']}")
                    st.write(f"✅ **Levenshtein-1 Matches:** {', '.join(result['levenshtein_1_matches']) if result['levenshtein_1_matches'] else 'None'}")
                    st.write(f"✅ **Levenshtein-2 Matches:** {', '.join(result['levenshtein_2_matches']) if result['levenshtein_2_matches'] else 'None'}")
                    st.write(f"🖥️ **Start Time:** `{result['start_time']}` ⏳ **End Time:** `{result['end_time']}`")
                    st.write(f"⏱️ **Processing Time:** `{result['execution_time']:.4f} seconds`")

                st.success(f"✅ Processing completed in `{total_time:.4f} seconds`!")
                st.write(f"**🔄 Time Before Multiprocessing:** ⏳ `{pre_time}`")
                st.write(f"**✅ Time After Multiprocessing:** ⏳ `{post_time}`")

if __name__ == "__main__":
    app = OCRTextCorrection()
    app.run()