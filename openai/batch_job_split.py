import os
import base64
import json
from pydantic import BaseModel
from typing import List, Optional

# --- Definice struktury (stejná) ---
class Chapter(BaseModel):
    model_config = {"extra": "forbid"}
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
    model_config = {"extra": "forbid"}
    chapters: List[Chapter]

INPUT_DIR = "img_trim"
OUTPUT_BATCH_DIR = "batch_files" # Složka pro JSONL soubory
MODEL = "gpt-4o"
CHUNK_SIZE = 80

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

def make_strict_schema(schema_dict):
    """Rekurzivně připraví schéma pro OpenAI Strict Mode."""
    if isinstance(schema_dict, dict):
        if schema_dict.get("type") == "object":
            schema_dict["additionalProperties"] = False
            # OpenAI vyžaduje, aby VŠECHNY klíče z 'properties' byly v 'required'
            if "properties" in schema_dict:
                schema_dict["required"] = list(schema_dict["properties"].keys())
        
        for value in schema_dict.values():
            make_strict_schema(value)
    elif isinstance(schema_dict, list):
        for item in schema_dict:
            make_strict_schema(item)
    return schema_dict

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

os.makedirs(OUTPUT_BATCH_DIR, exist_ok=True)

image_extensions = ('.jpg', '.jpeg', '.png')
all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(image_extensions)])
print(f"Celkem nalezeno {len(all_files)} obrázků.")

for i in range(0, len(all_files), CHUNK_SIZE):
    chunk = all_files[i:i + CHUNK_SIZE]
    batch_num = (i // CHUNK_SIZE) + 1
    jsonl_filename = os.path.join(OUTPUT_BATCH_DIR, f"batch_req_{batch_num}.jsonl")
    
    # Vygenerujeme základní schéma z Pydanticu
    raw_schema = OCRResponse.model_json_schema()
    # Upravíme ho, aby splňovalo OpenAI 'Strict' podmínky
    strict_schema = make_strict_schema(raw_schema)

    print(f"Generuji: {jsonl_filename} ({len(chunk)} souborů)")
    
    with open(jsonl_filename, "w", encoding="utf-8") as f:
        for filename in chunk:
            img_path = os.path.join(INPUT_DIR, filename)
            base64_image = encode_image(img_path)
            
            request_line = {
                "custom_id": filename,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
                            ]
                        }
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ocr_response",
                            "strict": True,
                            "schema": strict_schema
                        }
                    }
                }
            }
            f.write(json.dumps(request_line) + "\n")

print(f"\nHotovo! Všechny soubory jsou ve složce '{OUTPUT_BATCH_DIR}'.")