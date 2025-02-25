from ollama_ocr import OCRProcessor

ocr = OCRProcessor(model_name='llama3.2-vision:11b') 

result = ocr.process_image(
    image_path="C:/Users/Harihara Sudhan N/Downloads/image3.jpg",
    format_type="markdown",  # Options: markdown, text, json, structured, key_value
    prompt="Extract all text, focusing on medicine and Investigations." # Optional custom prompt
)
print(result)