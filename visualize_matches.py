import argparse
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
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
DEFAULT_FONT_SIZE = 32

# High-Contrast Palette by Level (RGB for Pillow)
# Level 0: Classic bright palette
# Level 1: Muted/Different hue palette
# Level 2+: Greyscale or very dark
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="High-contrast hierarchical visualization")
    parser.add_argument("-j", "--json_dir", type=str, default=DEFAULT_JSON_DIR)
    parser.add_argument("-i", "--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()

def get_level_color(field_name, level):
    """
    Selects color based on hierarchy depth level. [cite: 2026-02-26]
    """
    # Use level 1 palette for anything level 1 or deeper, but dim it further [cite: 2026-02-26]
    palette_idx = min(level, 1)
    base_color = HIERARCHY_PALETTE[palette_idx].get(field_name, (0, 0, 0))
    
    if level > 1:
        # Dim subsequent levels by 40% [cite: 2026-02-26]
        factor = 0.6 ** (level - 1)
        return tuple(int(c * factor) for c in base_color)
    
    return base_color

def draw_chapter_recursive(draw_obj, font, chapter, level=0):
    """
    Draws chapters with significant color shifts based on depth. [cite: 2026-02-26]
    """
    fields = ["name", "chapter_number", "page_number", "description"]
    centers = []
    
    # Visual thickness hierarchy [cite: 2026-02-26]
    current_width = 4 if level == 0 else 2

    for field in fields:
        bbox_key = f"{field}_bbox"
        bbox = chapter.get(bbox_key)
        field_text = chapter.get(field) 
        
        if bbox and len(bbox) == 2:
            p1 = tuple(map(int, bbox[0]))
            p2 = tuple(map(int, bbox[1]))
            centers.append(((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2))
            
            # Get color from the high-contrast palette [cite: 2026-02-26]
            field_color = get_level_color(field, level)
            
            # 1. Bounding box [cite: 2026-02-26]
            draw_obj.rectangle([p1, p2], outline=field_color, width=current_width)
            
            # 2. Text rendering [cite: 2026-02-26]
            if field_text:
                # Add "Level" prefix to the name field to make it ultra-obvious [cite: 2026-02-26]
                prefix = f"[{level}] " if field == "name" else ""
                draw_obj.text((p1[0] + 3, p1[1] + 2), prefix + str(field_text), font=font, fill=field_color)

    # Connecting lines with level-specific colors [cite: 2026-02-26]
    if len(centers) > 1:
        line_color = get_level_color("line", level)
        centers.sort(key=lambda p: p[0])
        for i in range(len(centers) - 1):
            draw_obj.line([centers[i], centers[i+1]], fill=line_color, width=current_width)

    # Increase level for recursion [cite: 2026-02-26]
    for sub in chapter.get("subchapters", []):
        draw_chapter_recursive(draw_obj, font, sub, level + 1)

def main():
    args = parse_arguments()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    try:
        font = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE)
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
        
        # Pillow conversion [cite: 2026-02-26]
        pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        draw_obj = ImageDraw.Draw(pil_img)

        for entry in data:
            draw_chapter_recursive(draw_obj, font, entry, level=0)
            
        # Write back to BGR for OpenCV [cite: 2026-02-26]
        final_res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_dir, base_name + "_vis.jpg"), final_res)
        print(f"Done: {base_name}")

if __name__ == "__main__":
    main()