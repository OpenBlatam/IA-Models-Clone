import os
import json
import fitz  # pymupdf
import re

PAPER_DIR = r"agents\backend\onyx\server\features\validacion_psicologica_ai\papers"
OUTPUT_FILE = r"agents\backend\onyx\server\features\validacion_psicologica_ai\data\training_data.jsonl"

def clean_text(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return clean_text(text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if not os.path.exists(PAPER_DIR):
        print(f"Paper directory not found: {PAPER_DIR}")
        return

    count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for filename in os.listdir(PAPER_DIR):
            if filename.lower().endswith('.pdf'):
                print(f"Processing {filename}...")
                pdf_path = os.path.join(PAPER_DIR, filename)
                extracted_text = process_pdf(pdf_path)
                
                if extracted_text and len(extracted_text) > 100: # Min length check
                    record = {
                        "text": extracted_text,
                        "metadata": {
                            "source": filename
                        }
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                    count += 1
    
    print(f"Processed {count} papers. Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
