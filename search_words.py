import re
import spacy
import nltk
from nltk.corpus import words
from difflib import get_close_matches
from Levenshtein import distance as lev

# Load NLP model and word list
nlp = spacy.load("en_core_web_sm")
nltk.download("words")
word_list = set(words.words())

def find_keyword_matches(text, keyword, context_window=50, max_distance=1):
    doc = nlp(text)
    words_in_text = set([token.text for token in doc if token.is_alpha])
    
    # Check if the keyword exists in the document
    keyword_in_text = keyword in words_in_text

    # Find closest matches using difflib
    closest_matches = get_close_matches(keyword, words_in_text, n=5, cutoff=0.7)

    # Find words within Levenshtein distance 1
    levenshtein_matches = [word for word in words_in_text if lev(keyword, word) <= max_distance]

    # Spell check suggestions (only if keyword is NOT in text)
    spell_suggestions = [w for w in word_list if lev(keyword, w) <= max_distance] if not keyword_in_text else []

    # Collect all possible keywords to search
    if keyword_in_text:
        search_terms = {keyword}  # Only exact matches if keyword is correct
    else:
        search_terms = set([keyword] + closest_matches + levenshtein_matches + spell_suggestions)

    matches = []
    for match in search_terms:
        for m in re.finditer(r'\b' + re.escape(match) + r'\b', text, re.IGNORECASE):
            start, end = m.start(), m.end()
            context_start = max(0, start - context_window)
            context_end = min(len(text), end + context_window)
            context = text[context_start:context_end]
            matches.append((match, context))

    return matches

# Example usage
document_text = """Your long document text goes here... It contains various words like machine, learning, artificial, intelligence, etc."""
keyword = "learing"# Correctly spelled keyword

results = find_keyword_matches(document_text, keyword)

for match, context in results:
    print(f"Found: {match}\nContext: {context}\n---")
