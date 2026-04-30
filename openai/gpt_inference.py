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
MODEL = "gpt-4o"
client = OpenAI()

def encode_image(image_path):
    """Zakóduje obrázek do base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def extract_toc_structured(image_path):
    """Pošle jeden obrázek do OpenAI."""
    base64_image = encode_image(image_path)

    # Prompt využívá normalizovaný grid 0-1000 pro vyšší přesnost
    PROMPT = """Analyze this scanned book table of contents page for visual grounding.
    Extract every entry into the provided JSON structure.

    Coordinate System:
    - Use a normalized grid of 0-1000.
    - Origin (0,0) is the top-left corner.
    - Return coordinates as [top-left, bottom-right] integers in [x, y] order: [[x1, y1], [x2, y2]].

    Extraction Rules:
    - "chapter_number": Extract ONLY the leading indicator (e.g., "1.", "I."). Must be separate from the name.
    - "name": The actual title, EXCLUDING the chapter number.
    - "page_number": The page number as a string.
    - "subchapters": Nest indented items here.

    Accuracy: Bounding boxes must tightly fit the text.
    """

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
        print(f"  [!] Chyba při volání API pro {os.path.basename(image_path)}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="OpenAI OCR - Batch Image Processing")
    parser.add_argument("-i", "--input", type=str, required=True, help="Cesta k obrázku nebo složce s obrázky")
    parser.add_argument("-o", "--output_dir", type=str, default="out", help="Složka pro uložení výsledků")
    
    args = parser.parse_args()

    # Identifikace souborů ke zpracování
    image_extensions = ('.jpg', '.jpeg', '.png')
    files_to_process = []

    if os.path.isdir(args.input):
        # Pokud je vstup složka, najdeme všechny obrázky
        files_to_process = [
            os.path.join(args.input, f) for f in os.listdir(args.input) 
            if f.lower().endswith(image_extensions)
        ]
        files_to_process.sort() # Seřazení podle názvu
        print(f"Nalezeno {len(files_to_process)} obrázků ve složce.")
    elif os.path.isfile(args.input) and args.input.lower().endswith(image_extensions):
        files_to_process = [args.input]
    else:
        print(f"Chyba: Vstup '{args.input}' není platný obrázek ani složka.")
        return

    # Vytvoření výstupní složky
    os.makedirs(args.output_dir, exist_ok=True)

    # Hlavní smyčka zpracování
    for i, img_path in enumerate(files_to_process, 1):
        filename = os.path.basename(img_path)
        print(f"[{i}/{len(files_to_process)}] Zpracovávám: {filename}")
        
        chapters = extract_toc_structured(img_path)
        
        if chapters:
            output_data = [c.model_dump() for c in chapters]
            
            # Generování názvu JSONu
            json_name = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(args.output_dir, json_name)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            print(f"  [OK] Uloženo do: {output_path}")
        else:
            print(f"  [SKIP] Nepodařilo se zpracovat {filename}")

    print("\nHotovo! Všechny soubory byly zpracovány.")

if __name__ == "__main__":
    main()