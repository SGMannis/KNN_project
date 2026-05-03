import json
import os
import argparse
from pathlib import Path
from PIL import Image

def normalize_bbox(bbox, img_w, img_h):
    """Převede [[x1, y1], [x2, y2]] z pixelů na 0-1000."""
    if not bbox or len(bbox) != 2:
        return bbox
    
    # Výpočet: (pixel / rozměr) * 1000
    p1 = [
        round((bbox[0][0] * 1000) / img_w),
        round((bbox[0][1] * 1000) / img_h)
    ]
    p2 = [
        round((bbox[1][0] * 1000) / img_w),
        round((bbox[1][1] * 1000) / img_h)
    ]
    
    # Ošetření přetečení (clamping) pro jistotu
    p1 = [max(0, min(1000, x)) for x in p1]
    p2 = [max(0, min(1000, x)) for x in p2]
    
    return [p1, p2]

def process_chapters_recursive(chapters, img_w, img_h):
    """Rekurzivně projde kapitoly a normalizuje všechny bbox pole."""
    bbox_fields = ["name_bbox", "chapter_number_bbox", "page_number_bbox", "description_bbox"]
    
    for chap in chapters:
        # Normalizace polí na aktuální úrovni
        for field in bbox_fields:
            if field in chap and chap[field] is not None:
                chap[field] = normalize_bbox(chap[field], img_w, img_h)
        
        # Rekurze pro podkapitoly
        if "subchapters" in chap and chap["subchapters"]:
            process_chapters_recursive(chap["subchapters"], img_w, img_h)

def main():
    parser = argparse.ArgumentParser(description="Normalizace souřadnic JSONů na 0-1000.")
    parser.add_argument("-i", "--image_dir", required=True, help="Složka s obrázky")
    parser.add_argument("-j", "--json_dir", required=True, help="Složka se vstupními JSONy")
    parser.add_argument("-o", "--output_dir", required=True, help="Složka pro uložené normalizované JSONy")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    json_files = list(Path(args.json_dir).glob("*.json"))
    print(f"Nalezeno {len(json_files)} JSON souborů.")

    processed_count = 0
    for json_path in json_files:
        # Hledání odpovídajícího obrázku (zkoušíme .jpg, .jpeg, .png)
        img_found = False
        img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
        
        for ext in img_extensions:
            img_path = Path(args.image_dir) / (json_path.stem + ext)
            if img_path.exists():
                img_found = True
                break
        
        if not img_found:
            print(f" [!] Varování: Obrázek pro {json_path.name} nebyl nalezen. Přeskakuji.")
            continue

        try:
            # Zjištění rozměrů obrázku
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            # Načtení JSONu
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Podpora pro oba formáty (s obalem 'chapters' i bez něj)
            if isinstance(data, dict) and "chapters" in data:
                process_chapters_recursive(data["chapters"], img_w, img_h)
            elif isinstance(data, list):
                process_chapters_recursive(data, img_w, img_h)
            else:
                print(f" [!] Chyba: Neznámý formát JSONu u {json_path.name}")
                continue

            # Uložení výsledku
            output_path = Path(args.output_dir) / json_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            processed_count += 1

        except Exception as e:
            print(f" [ERR] Selhalo zpracování {json_path.name}: {e}")

    print(f"\nHotovo! Úspěšně normalizováno {processed_count} souborů.")
    print(f"Výsledky uloženy v: {args.output_dir}")

if __name__ == "__main__":
    main()
