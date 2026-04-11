import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import json, os

MODEL_PATH = '/storage/brno2/home/xnehez01/KNN_project/models/Florence-2-large'
SCANS_DIR = '/storage/brno2/home/xnehez01/KNN_project/data/images'
OUT_DIR = '/storage/brno2/home/xnehez01/KNN_project/results/florence/json'

os.makedirs(OUT_DIR, exist_ok=True)

print('Načítavam Florence-2...')
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to('cuda')
print('Model načítaný!')

images = [f for f in os.listdir(SCANS_DIR) if f.endswith('.jpg')][:50]

for img_file in images:
    print(f'\nSpracovávam: {img_file}')
    image = Image.open(os.path.join(SCANS_DIR, img_file)).convert('RGB')
    inputs = processor(
    text='<OCR_WITH_REGION>',
    images=image,
    return_tensors='pt'
    ).to('cuda')

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            num_beams=3
        )

    result = processor.batch_decode(output, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        result,
        task='<OCR_WITH_REGION>',
        image_size=(image.width, image.height)
    )

    out_file = os.path.join(OUT_DIR, img_file.replace('.jpg', '_florence.json'))
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print(f'Výsledok: {str(parsed)[:300]}')
    print(f'Uložené: {out_file}')

print('\nVšetko hotovo!')