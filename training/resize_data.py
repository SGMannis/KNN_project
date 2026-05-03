import os
import json
from PIL import Image

IMAGE_DIR = '/storage/brno2/home/xnehez01/KNN_project/data/images'
JSONS_DIR = '/storage/brno2/home/xnehez01/KNN_project/data/jsons'
OUTPUT_IMAGE_DIR = '/storage/brno2/home/xnehez01/KNN_project/data/images_resized'
OUTPUT_JSONS_DIR = '/storage/brno2/home/xnehez01/KNN_project/data/jsons_resized'

MAX_IMAGE_SIDE = 1024
MAX_IMAGE_PIX = 1024 * 1024

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSONS_DIR, exist_ok=True)

def resize_img(image):
    pil = image.convert("RGB")
    w, h = pil.size
    scale_side = min(1.0, MAX_IMAGE_SIDE / float(max(w, h)))
    scale_area = (MAX_IMAGE_PIX / float(w * h)) ** 0.5 if (w * h) > MAX_IMAGE_PIX else 1.0
    scale = min(scale_side, scale_area)
    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil.resize((nw, nh), resample=Image.BICUBIC)
    return pil

def normalize_bbox(bbox, width, height):
    if bbox is None:
        return None
    return [
        [int(bbox[0][0] / width * 1000), int(bbox[0][1] / height * 1000)],
        [int(bbox[1][0] / width * 1000), int(bbox[1][1] / height * 1000)]
    ]

def normalize_item(item, width, height):
    item["name_bbox"] = normalize_bbox(item.get("name_bbox"), width, height)
    item["chapter_number_bbox"] = normalize_bbox(item.get("chapter_number_bbox"), width, height)
    item["page_number_bbox"] = normalize_bbox(item.get("page_number_bbox"), width, height)
    item["description_bbox"] = normalize_bbox(item.get("description_bbox"), width, height)
    if item.get("subchapters"):
        for sub in item["subchapters"]:
            normalize_item(sub, width, height)
    return item

image_files = os.listdir(IMAGE_DIR)
json_files = os.listdir(JSONS_DIR)
json_stems = [f.replace('.json', '') for f in json_files if f.endswith('.json')]

ok = 0
skip = 0

for filename in image_files:
    if not filename.endswith('.jpg'):
        continue
    stem = filename.replace('.jpg', '')
    if stem not in json_stems:
        skip += 1
        continue

    # Nacitaj obrazok
    img_path = os.path.join(IMAGE_DIR, filename)
    image = Image.open(img_path)
    width, height = image.size  # pôvodné rozmery pre normalizáciu

    # 1. Normalizuj bbox podľa PÔVODNÝCH rozmerov
    json_path = os.path.join(JSONS_DIR, stem + '.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    annotation = [normalize_item(item, width, height) for item in annotation]

    # 2. Zmenši obrazok
    image_resized = resize_img(image)
    image_resized.save(os.path.join(OUTPUT_IMAGE_DIR, filename))

    # 3. Uloz normalizovany JSON
    out_json_path = os.path.join(OUTPUT_JSONS_DIR, stem + '.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)

    ok += 1
    if ok % 100 == 0:
        print(f"Spracovanych: {ok}")

print(f"Hotovo! Spracovanych: {ok}, preskocených: {skip}")