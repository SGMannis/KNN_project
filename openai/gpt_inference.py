import os
import base64
import json
import argparse
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI

# --- Definice struktury (Pydantic) ---
class Chapter(BaseModel):
    name: Optional[str] = None
    chapter_number: Optional[str] = None
    page_number: Optional[str] = None
    description: Optional[str] = None
    name_bbox: Optional[List[List[int]]] = None
    chapter_number_bbox: Optional[List[List[int]]] = None
    page_number_bbox: Optional[List[List[int]]] = None
    description_bbox: Optional[List[List[int]]] = None
    subchapters: List['Chapter'] = []

class OCRResponse(BaseModel):
    chapters: List[Chapter]

# Nastavení modelu
MODEL = "gpt-4o-mini"
client = OpenAI()

def encode_image(image_path):
    """Zakóduje obrázek do base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def extract_toc_structured(image_path):
    """Pošle jeden obrázek do OpenAI."""
    base64_image = encode_image(image_path)

    PROMPT = """Analyze this scanned book table of contents page.
    Extract every entry into the provided JSON structure.

    Rules for fields:
    - "chapter_number": Extract ONLY the leading number or indicator (e.g., "1.", "I.", "A.", "Kapitola I."). If it's part of the line, it MUST be separated from the name.
    - "name": The actual title of the chapter, EXCLUDING the chapter number.
    - "page_number": The page number as a string.
    - "description": Any additional subtext or author name belonging to the entry.
    
    Hierarchy & Coordinates:
    1. Nest indented items under their parent chapter's 'subchapters' field.
    2. Provide ALL bounding boxes as [top-left, bottom-right] using normalized values (0-1000).
       - 0,0 is top-left corner.
       - 1000,1000 is bottom-right corner.
    3. Accuracy: Ensure bounding boxes for 'chapter_number' and 'name' are distinct and do not overlap if possible.
    
    IMPORTANT: If a line starts with a number (e.g. "1. Úvod"), "1." goes to chapter_number and "Úvod" goes to name."""

    try:
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"},
                        },
                    ],
                }
            ],
            response_format=OCRResponse,
        )
        return response.choices[0].message.parsed.chapters
    except Exception as e:
        print(f"Chyba při volání API: {e}")
        return None

def main():
    # Nastavení argumentů příkazové řádky
    parser = argparse.ArgumentParser(description="OpenAI OCR - Single Image Processing")
    parser.add_argument("-i", "--input", type=str, required=True, help="Cesta k obrázku k analýze")
    parser.add_argument("-o", "--output_dir", type=str, default="out", help="Složka pro uložení výsledků")
    
    args = parser.parse_args()

    # Kontrola, zda soubor existuje
    if not os.path.exists(args.input):
        print(f"Chyba: Soubor '{args.input}' nebyl nalezen.")
        return

    print(f"Zpracovávám: {args.input}")
    chapters = extract_toc_structured(args.input)
    
    if chapters:
        # Převedeme Pydantic objekty na JSON
        output_data = [c.model_dump() for c in chapters]
        
        # Vytvoření výstupní složky
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Vygenerování názvu JSONu podle původního obrázku
        filename = os.path.basename(args.input).replace(".jpg", ".json").replace(".jpeg", ".json").replace(".png", ".json")
        output_path = os.path.join(args.output_dir, filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
            
        print(f"Hotovo! Uloženo do: {output_path}")
    else:
        print("Nepodařilo se získat data z API.")

if __name__ == "__main__":
    main()
