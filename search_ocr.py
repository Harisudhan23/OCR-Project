from Levenshtein import distance

def find_closest_word(ocr_text, search_word, max_distance=1):
    words = ocr_text.split()  # Tokenize the text into words
    closest_match = None

    for word in words:
        if distance(word, search_word) <= max_distance:
            closest_match = word
            break  # Since we need only the first closest match

    return closest_match

# Test cases
test_cases = [
    {"ocr_text": "Th1s is an examp1e document scannd for test1ng.", "search_word": "example", "expected": "examp1e"},
    {"ocr_text": "The qick brown fx jumps over the lazy dog.", "search_word": "quick", "expected": "qick"},
    {"ocr_text": "ocr technology improvess accuracy over timr.", "search_word": "time", "expected": "timr"},
    {"ocr_text": "Invoice numbr: 123456", "search_word": "number", "expected": "numbr"},
    {"ocr_text": "Artifcial Intellgence is evoving rapidly.", "search_word": "Intelligence", "expected": "Intellgence"},
]

# Running test cases
for i, test in enumerate(test_cases, 1):
    result = find_closest_word(test["ocr_text"], test["search_word"])
    print(f"Test Case {i}: Expected '{test['expected']}', Got '{result}' -> {'✅' if result == test['expected'] else '❌'}")