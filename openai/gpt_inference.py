import os
import base64
import json
from typing import List, Optional, Tuple
from pydantic import BaseModel
from openai import OpenAI

# Definice struktury (přesně podle tvého zadání) [cite: 2026-02-26]
class Chapter(BaseModel):
    name: Optional[str] = None
    chapter_number: Optional[str] = None
    page_number: Optional[str] = None
    description: Optional[str] = None
    # Pydantic v Structured Outputs preferuje List před Tuple
    name_bbox: Optional[List[List[int]]] = None
    chapter_number_bbox: Optional[List[List[int]]] = None
    page_number_bbox: Optional[List[List[int]]] = None
    description_bbox: Optional[List[List[int]]] = None
    subchapters: List['Chapter'] = []

class OCRResponse(BaseModel):
    chapters: List[Chapter]

# Cesta k obrázku
TEST_IMAGE_PATH = "../data/project-38-at-2026-02-24-13-47-846604c7/images/ff92e831-d4e2-43e2-8da8-444925b5f764.c9ee937d-f9a3-11e6-b230-00155d012102.287.None.svkhk.jpg"
MODEL = "gpt-4o-mini"

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def extract_toc_structured(image_path):
    base64_image = encode_image(image_path)

    # Prompt upravený pro Structured Outputs a normalizaci [cite: 2026-02-26]
    PROMPT = """Analyze this table of contents. 
    1. Extract all chapters and subchapters.
    2. Hierarchy: Nest indented items under their parent chapter's 'subchapters' field.
    3. Coordinates: Provide ALL bounding boxes as [top-left, bottom-right] using normalized values (0-1000).
       - 0,0 is top-left corner.
       - 1000,1000 is bottom-right corner.
    4. Accuracy: Ensure bounding boxes tightly fit the text."""

    try:
        # Použijeme .beta.chat.completions.parse pro automatické parsování [cite: 2026-02-26]
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
            response_format=OCRResponse, # Tady nutíme model dodržet tvou strukturu [cite: 2026-02-26]
        )

        # Vrátíme jen čistý list kapitol pro tvou vizualizaci
        return response.choices[0].message.parsed.chapters

    except Exception as e:
        print(f"Chyba: {e}")
        return None

if __name__ == "__main__":
    if os.path.exists(TEST_IMAGE_PATH):
        print(f"Odesílám {TEST_IMAGE_PATH}...")
        chapters = extract_toc_structured(TEST_IMAGE_PATH)
        
        if chapters:
            # Převedeme Pydantic objekty na čistý JSON list [cite: 2026-02-26]
            output_data = [c.model_dump() for c in chapters]
            
            output_dir = "out"
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(TEST_IMAGE_PATH).replace(".jpg", ".json")
            
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            print(f"Hotovo! Uloženo do {output_dir}/{filename}")