import streamlit as st
import os
import re
import multiprocessing
from datetime import datetime
from Levenshtein import distance as levenshtein_distance

class OCRTextCorrection:
    def __init__(self, num_cores):
        self.ocr_text = ""
        self.search_words = []
        self.num_cores = num_cores

    @staticmethod
    def clean_word(word, preserve_spaces=False, lowercase_only=False):
        word = re.sub(r'\s+', ' ', word).strip()
        return re.sub(r'[^\w\s]', '', word.lower()) if preserve_spaces else re.sub(r'[^\w]', '', word.lower())

    @staticmethod
    def sanitize_match(word):
        return re.sub(r'[:;,!?]+$', '', word).strip()

    @staticmethod
    def process_search_word(args):
        core_id, search_word, words = args
        start_time = datetime.now()
        search_clean = OCRTextCorrection.clean_word(search_word, preserve_spaces=True, lowercase_only=True)
        best_match, best_distance = None, float('inf')
        lev_1_matches, lev_2_matches = set(), set()

        for window_size in range(1, 4):
            for i in range(len(words) - window_size + 1):
                candidate = " ".join(words[i:i + window_size])
                candidate_clean = OCRTextCorrection.clean_word(candidate, preserve_spaces=True, lowercase_only=True)
                dist = levenshtein_distance(candidate_clean, search_clean)
                if dist == 1:
                    lev_1_matches.add(OCRTextCorrection.sanitize_match(candidate))
                elif dist == 2:
                    lev_2_matches.add(OCRTextCorrection.sanitize_match(candidate))
                if dist < best_distance:
                    best_match, best_distance = OCRTextCorrection.sanitize_match(candidate), dist

        end_time = datetime.now()
        return search_word, {
            "closest_match": best_match or 'N/A',
            "levenshtein_1_matches": sorted(lev_1_matches),
            "levenshtein_2_matches": sorted(lev_2_matches),
            "start_time": start_time.strftime('%H:%M:%S.%f'),
            "end_time": end_time.strftime('%H:%M:%S.%f'),
            "execution_time": (end_time - start_time).total_seconds()
        }

    def find_closest_matches(self, ocr_text, search_words):
        words = ocr_text.split()
        num_cores = min(self.num_cores, os.cpu_count())
        pre_time = datetime.now()

        with multiprocessing.Pool(processes=num_cores) as pool:
            results_list = pool.map(OCRTextCorrection.process_search_word, [(i, word, words) for i, word in enumerate(search_words)])

        post_time = datetime.now()
        return dict(results_list), pre_time.strftime('%H:%M:%S.%f'), post_time.strftime('%H:%M:%S.%f'), (post_time - pre_time).total_seconds()


class OCRTextCorrectionApp:
    def __init__(self):
        self.num_cores = os.cpu_count()
        self.processor = OCRTextCorrection(num_cores=self.num_cores)

    def run(self):
        st.title("🔍 OCR Text Correction and Analysis (Optimized Multiprocessing)")
        self.processor.ocr_text = st.text_area("📄 Enter OCR Text (2000+ words recommended):", height=300)
        search_words_input = st.text_input("🔎 Enter Search Words (comma-separated):")
        self.processor.num_cores = st.slider("⚙️ Select Number of Cores:", 1, os.cpu_count(), 1)

        if st.button("🔍 Find Matches"):
            if not self.processor.ocr_text or not search_words_input:
                st.warning("Please enter OCR text and search words! ⚠️")
            else:
                search_words = [word.strip() for word in search_words_input.split(",")]
                results, pre_time, post_time, total_time = self.processor.find_closest_matches(
                    self.processor.ocr_text, search_words
                )
                
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
    app = OCRTextCorrectionApp()
    app.run()
