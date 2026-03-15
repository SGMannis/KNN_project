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

# Path to a font that supports Czech characters.
# FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
DEFAULT_FONT_SIZE = 32

# High-Contrast Palette by Level (RGB for Pillow)
HIERARCHY_PALETTE = {
    0: {
        "name": (255, 0, 0),           # Bright Red
        "chapter_number": (0, 0, 255),  # Bright Blue
        "page_number": (0, 255, 0),     # Bright Green
        "description": (255, 255, 0),   # Yellow
        "line": (0, 0, 0)               # Black
    },
    1: {
        "name": (128, 0, 128),         # Purple
        "chapter_number": (255, 165, 0),# Orange
        "page_number": (0, 128, 128),   # Teal
        "description": (165, 42, 42),   # Brown
        "line": (100, 100, 100)         # Gray
    }
}

def get_system_font():
    system = platform.system()
    
    if system == "Darwin":  # macOS
        paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf"
        ]
    elif system == "Windows":
        paths = [
            "C:\\Windows\\Fonts\\arial.ttf",
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf")
        ]
    else:  # Linux (Common paths for Arial or DejaVu)
        paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]

    for path in paths:
        if os.path.exists(path):
            return path
    return None

def parse_arguments():
    parser = argparse.ArgumentParser(description="Side-by-side hierarchical visualization with granular control")
    
    # Path arguments
    parser.add_argument("-j", "--json_dir", type=str, default=DEFAULT_JSON_DIR, help="JSON annotations directory")
    parser.add_argument("-i", "--image_dir", type=str, default=DEFAULT_IMAGE_DIR, help="Source images directory")
    parser.add_argument("-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    
    # Original Page options (Left side)
    parser.add_argument("--p_bbox", action="store_true", default=True, help="Show bounding boxes on original page")
    parser.add_argument("--p_text", action="store_true", default=False, help="Show text on original page")
    parser.add_argument("--p_line", action="store_true", default=True, help="Show alignment lines on original page")
    
    # Extension Page options (Right side)
    parser.add_argument("--e_bbox", action="store_true", default=False, help="Show bounding boxes on extension")
    parser.add_argument("--e_text", action="store_true", default=True, help="Show text on extension")
    parser.add_argument("--e_line", action="store_true", default=False, help="Show alignment lines on extension")
    
    return parser.parse_args()

def get_level_color(field_name, level):
    palette_idx = min(level, 1)
    base_color = HIERARCHY_PALETTE[palette_idx].get(field_name, (0, 0, 0))
    if level > 1:
        factor = 0.6 ** (level - 1)
        return tuple(int(c * factor) for c in base_color)
    return base_color

def draw_chapter_recursive(draw_obj, font, chapter, x_offset, args, level=0):
    """
    Draws chapters based on granular visibility flags. [cite: 2026-02-26]
    """
    fields = ["name", "chapter_number", "page_number", "description"]
    centers_left = []
    centers_right = []
    
    current_width = 4 if level == 0 else 2

    for field in fields:
        bbox_key = f"{field}_bbox"
        bbox = chapter.get(bbox_key)
        field_text = chapter.get(field) 
        
        if bbox and len(bbox) == 2:
            p1_l = (int(bbox[0][0]), int(bbox[0][1]))
            p2_l = (int(bbox[1][0]), int(bbox[1][1]))
            centers_left.append(((p1_l[0] + p2_l[0]) // 2, (p1_l[1] + p2_l[1]) // 2))

            p1_r = (p1_l[0] + x_offset, p1_l[1])
            p2_r = (p2_l[0] + x_offset, p2_l[1])
            centers_right.append(((p1_r[0] + p2_r[0]) // 2, (p1_r[1] + p2_r[1]) // 2))
            
            field_color = get_level_color(field, level)
            
            # 1. Bounding Boxes [cite: 2026-02-26]
            if args.p_bbox:
                draw_obj.rectangle([p1_l, p2_l], outline=field_color, width=current_width)
            if args.e_bbox:
                draw_obj.rectangle([p1_r, p2_r], outline=field_color, width=current_width)
            
            # 2. Text [cite: 2026-02-26]
            if field_text:
                prefix = f"[{level}] " if field == "name" else ""
                txt = prefix + str(field_text)
                if args.p_text:
                    draw_obj.text((p1_l[0] + 3, p1_l[1] + 2), txt, font=font, fill=field_color)
                if args.e_text:
                    draw_obj.text((p1_r[0] + 3, p1_r[1] + 2), txt, font=font, fill=field_color)

    # 3. Alignment Lines [cite: 2026-02-26]
    line_color = get_level_color("line", level)
    if len(centers_left) > 1:
        centers_left.sort(key=lambda p: p[0])
        centers_right.sort(key=lambda p: p[0])
        for i in range(len(centers_left) - 1):
            if args.p_line:
                draw_obj.line([centers_left[i], centers_left[i+1]], fill=line_color, width=current_width)
            if args.e_line:
                draw_obj.line([centers_right[i], centers_right[i+1]], fill=line_color, width=current_width)

    for sub in chapter.get("subchapters", []):
        draw_chapter_recursive(draw_obj, font, sub, x_offset, args, level + 1)

def main():
    args = parse_arguments()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    font_path = get_system_font()
    try:
        if font_path:
            font = ImageFont.truetype(font_path, DEFAULT_FONT_SIZE)
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    for json_path in glob.glob(os.path.join(args.json_dir, "*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = os.path.join(args.image_dir, base_name + ".jpg")
        
        if not os.path.exists(img_path): continue
        orig_img = cv2.imread(img_path)
        if orig_img is None: continue
        
        h, w, _ = orig_img.shape
        # Create extended canvas [cite: 2026-02-26]
        canvas = np.full((h, w * 2, 3), 255, dtype=np.uint8)
        canvas[:h, :w] = orig_img
        
        # Pillow conversion [cite: 2026-02-26]
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw_obj = ImageDraw.Draw(pil_img)

        for entry in data:
            draw_chapter_recursive(draw_obj, font, entry, x_offset=w, args=args, level=0)
            
        final_res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_dir, base_name + "_vis.jpg"), final_res)
        print(f"Done: {base_name}")

if __name__ == "__main__":
    main()
