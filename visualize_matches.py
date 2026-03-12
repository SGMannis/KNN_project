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

# Color definition (Stored as BGR, but we will convert to RGB for Pillow)
COLORS = {
    "name": (0, 0, 255),           # red
    "chapter_number": (255, 0, 0), # blue
    "page_number": (0, 255, 0),    # green
    "description": (0, 255, 255),  # yellow
    "line": (0, 0, 0)              # black
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualization of updated matched output with alignment lines")

    parser.add_argument(
        "-j", "--json_dir", 
        type=str,
        default=DEFAULT_JSON_DIR,
        help="path to the fully matched json directory"
    )

    parser.add_argument(
        "-i", "--image_dir",
        type=str,
        default=DEFAULT_IMAGE_DIR, 
        help="path to the source image directory"
    )

    parser.add_argument(
        "-o", "--output_dir", 
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="path to the output directory"
    )

    return parser.parse_args()

def draw_chapter_recursive(draw_obj, font, chapter):
    """
    Recursively draws elements and their internal text using Pillow. [cite: 2026-02-26]
    """
    fields = ["name", "chapter_number", "page_number", "description"]
    centers = []

    # BGR to RGB helper for Pillow [cite: 2026-02-26]
    def bgr_to_rgb(bgr): return (bgr[2], bgr[1], bgr[0])

    for field in fields:
        bbox_key = f"{field}_bbox"
        bbox = chapter.get(bbox_key)
        field_text = chapter.get(field) 
        
        if bbox and len(bbox) == 2:
            p1 = tuple(map(int, bbox[0]))
            p2 = tuple(map(int, bbox[1]))
            
            centers.append(((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2))
            
            field_color = bgr_to_rgb(COLORS.get(field, (255, 255, 255)))
            
            # 1. Draw the bounding box [cite: 2026-02-26]
            draw_obj.rectangle([p1, p2], outline=field_color, width=2)
            
            # 2. Draw the text inside the box [cite: 2026-02-26]
            if field_text:
                draw_obj.text((p1[0] + 3, p1[1] + 2), str(field_text), font=font, fill=field_color)

    # Horizontal alignment lines [cite: 2026-02-26]
    if len(centers) > 1:
        centers.sort(key=lambda p: p[0])
        for i in range(len(centers) - 1):
            draw_obj.line([centers[i], centers[i+1]], fill=bgr_to_rgb(COLORS["line"]), width=2)

    # Recurse into subchapters [cite: 2026-02-26]
    for sub in chapter.get("subchapters", []):
        draw_chapter_recursive(draw_obj, font, sub)

def main():
    args = parse_arguments()
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    # Load font once [cite: 2026-02-26]
    try:
        font = ImageFont.truetype(FONT_PATH, 16)
    except:
        font = ImageFont.load_default()

    json_files = glob.glob(os.path.join(args.json_dir, "*.json"))
    
    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = os.path.join(args.image_dir, base_name + ".jpg")
        
        if not os.path.exists(img_path): continue
        orig_img = cv2.imread(img_path)
        if orig_img is None: continue
        
        # 1. Single conversion to Pillow RGB [cite: 2026-02-26]
        pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        draw_obj = ImageDraw.Draw(pil_img)

        # 2. Process all drawing operations [cite: 2026-02-26]
        for entry in data:
            draw_chapter_recursive(draw_obj, font, entry)
            
        # 3. Final conversion back to OpenCV BGR [cite: 2026-02-26]
        final_res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(args.output_dir, base_name + "_vis.jpg"), final_res)
        print(f"Done: {base_name}")

if __name__ == "__main__":
    main()
