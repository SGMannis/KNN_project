import os
import base64
import json
from openai import OpenAI

TEST_IMAGE_PATH = "../data/project-38-at-2026-02-24-13-47-846604c7/images/ff92e831-d4e2-43e2-8da8-444925b5f764.c9ee937d-f9a3-11e6-b230-00155d012102.287.None.svkhk.jpg"
MODEL = "gpt-5.4-mini"

# Inicializace klienta (klíč už máš v proměnných prostředí)
client = OpenAI()

def encode_image(image_path):
    """Zakóduje obrázek do base64, aby mohl být poslán přes API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_toc_to_json(image_path):
    # Příprava obrázku
    base64_image = encode_image(image_path)

    # Tvůj vyladěný prompt (mírně upravený pro GPT-4o)
    PROMPT = """Analyze this scanned book table of contents page.
    Extract all entries and return a RAW JSON ARRAY of objects. 
    
    Each object in the array must have this structure:
    - "name": chapter title (string or null)
    - "chapter_number": e.g. "XXII.", "1.", "a)" (string or null)
    - "page_number": page number (string or null)
    - "description": subtext (string or null)
    - "name_bbox": [[x1, y1], [x2, y2]] (integers)
    - "chapter_number_bbox": [[x1, y1], [x2, y2]] or null
    - "page_number_bbox": [[x1, y1], [x2, y2]] or null
    - "description_bbox": [[x1, y1], [x2, y2]] or null
    - "subchapters": list of objects with the same structure (empty list [] if none)

    Rules:
    1. Return ONLY the JSON array (starting with [ and ending with ]).
    2. Hierarchy: Indented items belong to the "subchapters" of the entry above them.
    3. Use pixel coordinates for bboxes.
    4. Do not include markdown formatting like ```json.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high" # Důležité pro čtení malého textu a souřadnic
                            },
                        },
                    ],
                }
            ],
            response_format={ "type": "json_object" }, # Zajišťuje validní JSON výstup
            # max_tokens=4096
        )

        # Extrakce textu z odpovědi
        output_text = response.choices[0].message.content
        data = json.loads(output_text)

        # Logika pro "rozbalení" listu:
        if isinstance(data, dict):
            # Pokud je v objektu jen jeden klíč a jeho obsahem je list, vezmeme ten list
            # To pokryje "result", "chapters", "entries" atd.
            for key in data:
                if isinstance(data[key], list):
                    return data[key]
        
        return data # Pokud už je to list, vrátí ho přímo

    except Exception as e:
        print(f"Error during API call: {e}")
        return None

# TEST NA JEDNOM OBRÁZKU
image_to_test = TEST_IMAGE_PATH
if os.path.exists(image_to_test):
    print(f"Odesílám {image_to_test} do {MODEL}...")
    result = extract_toc_to_json(image_to_test)
    
    if result:
        # Uložení výsledku
        output_dir = os.path.join(os.path.dirname(__file__), "out")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{os.path.splitext(os.path.basename(image_to_test))[0]}.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Hotovo! Výsledek je v {output_path}")
else:
    print("Obrázek nenalezen.")
