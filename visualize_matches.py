import argparse
import platform
import cv2
import json
import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Default Path configuration
DEFAULT_JSON_DIR = "out/"
DEFAULT_IMAGE_DIR = "data/project-38-at-2026-02-24-13-47-846604c7/images/"
DEFAULT_OUTPUT_DIR = "out_vis/"
COLOR_WARNING = (255, 0, 255) 

# Global counters/trackers
missing_counter = 0
current_file_has_error = False 
DEFAULT_FONT_SIZE = 24

HIERARCHY_PALETTE = {
    0: {
        "name": (255, 0, 0), "chapter_number": (0, 0, 255), 
        "page_number": (0, 255, 0), "description": (255, 255, 0), "line": (0, 0, 0)
    },
    1: {
        "name": (128, 0, 128), "chapter_number": (255, 165, 0), 
        "page_number": (0, 128, 128), "description": (165, 42, 42), "line": (100, 100, 100)
    }
}

def get_system_font():
    system = platform.system()
    if system == "Darwin":
        paths = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"]
    elif system == "Windows":
        paths = ["C:\\Windows\\Fonts\\arial.ttf", os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf")]
    else:
        paths = ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for path in paths:
        if os.path.exists(path): return path
    return None

def wrap_text(text, font, max_width):
    words = str(text).split(' ')
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        if font.getlength(test_line) <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    return lines

def draw_wrapped_text(draw_obj, position, text, font, fill, max_width):
    lines = wrap_text(text, font, max_width)
    x, y = position
    line_height = font.getbbox("Ay")[3] + 2 
    for line in lines:
        draw_obj.text((x, y), line, font=font, fill=fill)
        y += line_height

def parse_arguments():
    parser = argparse.ArgumentParser(description="Side-by-side hierarchical visualization")
    parser.add_argument("-j", "--json_dir", type=str, default=DEFAULT_JSON_DIR)
    parser.add_argument("-i", "--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-v", "--verbose", action="store_true")
    # PŘIDANÝ ARGUMENT PRO NORMALIZACI
    parser.add_argument("--normalized", action="store_true", help="Považovat souřadnice za 0-1000 místo pixelů")
    
    parser.add_argument("--p_bbox", action="store_true", default=True)
    parser.add_argument("--p_text", action="store_true", default=False)
    parser.add_argument("--p_line", action="store_true", default=True)
    parser.add_argument("--e_bbox", action="store_true", default=False)
    parser.add_argument("--e_text", action="store_true", default=True)
    parser.add_argument("--e_line", action="store_true", default=False)
    return parser.parse_args()

def get_level_color(field_name, level):
    palette_idx = min(level, 1)
    base_color = HIERARCHY_PALETTE[palette_idx].get(field_name, (0, 0, 0))
    if level > 1:
        factor = 0.6 ** (level - 1)
        return tuple(int(c * factor) for c in base_color)
    return base_color

# PŘIDÁNY PARAMETRY img_w A img_h
def draw_chapter_recursive(draw_obj, font, chapter, x_offset, args, img_w, img_h, level=0):
    global missing_counter, current_file_has_error
    fields = ["name", "chapter_number", "page_number", "description"]
    centers_left = []
    centers_right = []
    
    for field in fields:
        bbox_key = f"{field}_bbox"
        bbox = chapter.get(bbox_key)
        field_text = chapter.get(field) 
        
        is_missing = field_text is None or (isinstance(field_text, str) and not field_text.strip())

        if bbox and len(bbox) == 2:
            if is_missing:
                missing_counter += 1
                current_file_has_error = True

            # LOGIKA PŘEPOČTU
            if args.normalized:
                p1_l = (int(bbox[0][0] * img_w / 1000), int(bbox[0][1] * img_h / 1000))
                p2_l = (int(bbox[1][0] * img_w / 1000), int(bbox[1][1] * img_h / 1000))
            else:
                p1_l = (int(bbox[0][0]), int(bbox[0][1]))
                p2_l = (int(bbox[1][0]), int(bbox[1][1]))
            
            box_w = max(1, p2_l[0] - p1_l[0])

            p1_r = (p1_l[0] + x_offset, p1_l[1])
            p2_r = (p2_l[0] + x_offset, p2_l[1])

            centers_left.append(((p1_l[0] + p2_l[0]) // 2, (p1_l[1] + p2_l[1]) // 2))
            centers_right.append(((p1_r[0] + p2_r[0]) // 2, (p1_r[1] + p2_r[1]) // 2))
            
            field_color = COLOR_WARNING if is_missing else get_level_color(field, level)
            rect_width = 6 if is_missing else (4 if level == 0 else 2)
            
            try:
                # Zkusíme nakreslit to, co model poslal (bez oprav)
                if args.p_bbox:
                    draw_obj.rectangle([p1_l, p2_l], outline=field_color, fill=None, width=rect_width)
                if args.e_bbox:
                    draw_obj.rectangle([p1_r, p2_r], outline=field_color, fill=None, width=rect_width)
            except ValueError:
                # LLM POSLAL ŠPATNÉ SOUŘADNICE (x1 < x0 nebo y1 < y0)
                current_file_has_error = True
                print(f" [!] BBox Error: p1={p1_l}, p2={p2_l} - Invalid rectangle.")
                
                # Nakreslíme aspoň varovný bod v místě p1, aby bylo vidět, kde se model "sekl"
                draw_obj.ellipse([p1_l[0]-5, p1_l[1]-5, p1_l[0]+5, p1_l[1]+5], fill=(255, 0, 0))
                draw_obj.text(p1_l, "BBOX ERROR", fill=(255, 0, 0), font=font)
            
            txt = f"[{level}] {field_text}" if field == "name" and not is_missing else str(field_text)
            if is_missing: txt = "MISSING"
            
            text_color = (0, 0, 0) if is_missing else field_color
            if args.p_text:
                draw_wrapped_text(draw_obj, (p1_l[0] + 3, p1_l[1] + 2), txt, font, text_color, box_w)
            if args.e_text:
                draw_wrapped_text(draw_obj, (p1_r[0] + 3, p1_r[1] + 2), txt, font, text_color, box_w)

    line_color = get_level_color("line", level)
    if len(centers_left) > 1:
        centers_left.sort(key=lambda p: p[0])
        centers_right.sort(key=lambda p: p[0])
        for i in range(len(centers_left) - 1):
            if args.p_line:
                draw_obj.line([centers_left[i], centers_left[i+1]], fill=line_color, width=2)
            if args.e_line:
                draw_obj.line([centers_right[i], centers_right[i+1]], fill=line_color, width=2)

    for sub in chapter.get("subchapters", []):
        draw_chapter_recursive(draw_obj, font, sub, x_offset, args, img_w, img_h, level + 1)

def main():
    global current_file_has_error
    print("="*30)
    print(f"VISUALIZATION SCRIPT")
    print("="*30+"\n")

    args = parse_arguments()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    font_path = get_system_font()
    try:
        font = ImageFont.truetype(font_path, DEFAULT_FONT_SIZE) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    json_files = glob.glob(os.path.join(args.json_dir, "*.json"))
    processed_count = 0

    print(f"Found {len(json_files)} JSON files. Processing...")
    for json_path in json_files:
        current_file_has_error = False 
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON {json_path}: {e}")
            continue
        
        # --- ROBUSTNÍ LOGIKA PRO HYBRIDNÍ FORMÁT ---
        # Pokud je data slovník a obsahuje klíč "chapters", vezmeme ten. 
        # Pokud je to už seznam, vezmeme ho přímo.
        if isinstance(data, dict) and "chapters" in data:
            entries_to_process = data["chapters"]
        elif isinstance(data, list):
            entries_to_process = data
        else:
            print(f"Warning: Unexpected JSON format in {json_path}. Skipping.")
            continue
        # ------------------------------------------

        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = os.path.join(args.image_dir, base_name + ".jpg")
        
        if not os.path.exists(img_path):
            if args.verbose: print(f"Image not found: {img_path}")
            continue

        orig_img = cv2.imread(img_path)
        if orig_img is None: continue
        
        h, w, _ = orig_img.shape
        canvas = np.full((h, w * 2, 3), 255, dtype=np.uint8)
        canvas[:h, :w] = orig_img
        
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw_obj = ImageDraw.Draw(pil_img)

        # Teď iterujeme přes entries_to_process, což je VŽDY seznam kapitol
        for entry in entries_to_process:
            draw_chapter_recursive(draw_obj, font, entry, x_offset=w, args=args, img_w=w, img_h=h, level=0)
            
        final_res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        suffix = "_error" if current_file_has_error else ""
        save_name = f"{base_name}_vis{suffix}.jpg"
        
        cv2.imwrite(os.path.join(args.output_dir, save_name), final_res)
        processed_count += 1
        
        if args.verbose:
            print(f"Done: {save_name}")

    print("\n" + "="*30)
    print(f"VISUALIZATION COMPLETE")
    print(f"Files processed: {processed_count}")
    print("="*30)

    print("\n" + "="*30)
    print(f"VISUALIZATION COMPLETE")
    print(f"Files processed: {processed_count}")
    print("="*30)

if __name__ == "__main__":
    main()