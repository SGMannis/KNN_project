import os
import json
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Cesty
MODEL_PATH = "/storage/brno2/home/xnehez01/KNN_project/models/Qwen3-VL-8B-Instruct"
SCANS_DIR  = "/storage/brno2/home/xnehez01/KNN_project/data/images"
OUTPUT_DIR = "/storage/brno2/home/xnehez01/KNN_project/results/qwen3_8b/json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Načítanie modelu
print("Načítavam model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model OK")

# Prompt
PROMPT = PROMPT = """Analyze this scanned book table of contents page.

Extract all entries and return a JSON list where each item has:

- "name": the chapter title text

- "chapter_number": chapter number if present (e.g. "1.", "a)", "I."), otherwise null

- "page_number": page number as string if present, otherwise null

- "description": descriptive subtext belonging to this entry if present, otherwise null

- "name_bbox": [[x1, y1], [x2, y2]] 

- "chapter_number_bbox": [[x1, y1], [x2, y2]] or null if no chapter number

- "page_number_bbox": [[x1, y1], [x2, y2]] or null if no page number

- "description_bbox": [[x1, y1], [x2, y2]] or null if no description

- "subchapters": list of subchapters in the same format, or empty list [] if none

Rules:

- If a chapter has indented entries below it, put them in "subchapters"

- Subchapters follow the same structure recursively

- If the table of contents is simple with no hierarchy, return a flat list with empty "subchapters"

- Preserve the reading order from top to bottom

IMPORTANT: Return ONLY the raw JSON array. Do NOT use markdown code blocks. Do NOT write ```json. Start your response directly with [ and end with ]."""
images = [f for f in os.listdir(SCANS_DIR) if f.endswith('.jpg')][:50]
print(f"Nájdených {len(images)} obrázkov")

# prejde obrazky a vytvori cestu
for img_file in images:
    img_path = os.path.join(SCANS_DIR, img_file)
    print(f"Spracúvam: {img_file}")

    # modelu poslem image a prompt pre kazdy tento obrazok
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": PROMPT}
            ]
        }
    ]

    # Príprava vstupu text do text formatu zo spravy vytiahne obrazok  a processor spoji vsetko do tensorov 
    # .to(model.device) presun na gpu
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt"
    ).to(model.device)

    # Generovanie torch.no_grad() vypne gradienty kedze len inferencia

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True, 
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.0,
        )

    # generated_ids ma v sebe aj vstup aj vystup zmazaem vstup 
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    #zase dostane obycajny text 
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Uloženie výstupu
    out_file = os.path.join(OUTPUT_DIR, img_file.replace('.jpg', '.json'))
    try:
        result = json.loads(output_text)
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  OK → {out_file}")
    except json.JSONDecodeError:
        # Ak model nevrátil čistý JSON, ulož raw výstup
        with open(out_file.replace('.json', '_raw.txt'), 'w') as f:
            f.write(output_text)
        print(f"  JSON parse chyba — uložené ako raw txt")

print("Hotovo!")