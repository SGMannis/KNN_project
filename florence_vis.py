from PIL import Image, ImageDraw, ImageFont
import json, os

RESULTS_DIR = '/storage/brno2/home/xnehez01/KNN_project/results/florence/json'
SCANS_DIR = '/storage/brno2/home/xnehez01/KNN_project/data/images'
VIS_DIR = '/storage/brno2/home/xnehez01/KNN_project/results//florence/vis'

os.makedirs(VIS_DIR, exist_ok=True)

for json_file in os.listdir(RESULTS_DIR):
    if not json_file.endswith('_florence.json'):
        continue

    img_file = json_file.replace('_florence.json', '.jpg')
    img_path = os.path.join(SCANS_DIR, img_file)
    json_path = os.path.join(RESULTS_DIR, json_file)

    if not os.path.exists(img_path):
        print(f'Obrázok nenájdený: {img_file}')
        continue

    image = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    with open(json_path, 'r') as f:
        data = json.load(f)

    ocr_data = data.get('<OCR_WITH_REGION>', {})
    quad_boxes = ocr_data.get('quad_boxes', [])
    labels = ocr_data.get('labels', [])

    for i, (quad, label) in enumerate(zip(quad_boxes, labels)):
        # quad = [x1,y1, x2,y2, x3,y3, x4,y4]
        x1, y1 = quad[0], quad[1]
        x3, y3 = quad[4], quad[5]

        draw.rectangle([
                min(x1, x3), min(y1, y3),
                max(x1, x3), max(y1, y3)
            ], outline='red', width=2)
        safe_label = label[:30].encode('ascii', 'ignore').decode('ascii')
        draw.text((x1, y1 - 15), safe_label, fill='red')

    out_path = os.path.join(VIS_DIR, img_file)
    image.save(out_path)
    print(f'Uložené: {out_path}')

print('Hotovo!')