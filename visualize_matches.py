import argparse
import cv2
import json
import os
import glob

# Path configuration
JSON_DIR = "out/"
IMAGE_DIR = "data/project-38-at-2026-02-24-13-47-846604c7/images/"
OUTPUT_DIR = "out_vis/"

# Color definition for separate elements.
COLORS = {
    "name": (0, 0, 255),           # red
    "chapter_number": (255, 0, 0), # blue
    "page_number": (0, 255, 0),    # green
    "description": (0, 255, 255)   # yellow
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualization of matched output")

    parser.add_argument(
        "-j", "--json_dir", 
        type=str,
        default=JSON_DIR,
        help="path to the fully matched json directory"
    )

    parser.add_argument(
        "-i", "--image_dir",
        type=str,
        default=IMAGE_DIR, 
        help="path to the source image directory"
    )

    parser.add_argument(
        "-o", "--output_dir", 
        type=str,
        default=OUTPUT_DIR,
        help="path to the output directory"
    )

    return parser.parse_args()

def draw_chapter(img, chapter):
    """Rekurzivně vykreslí kapitolu a její podkapitoly."""
    fields_order = ["name", "chapter_number", "page_number", "description"]
    polygon = chapter.get("polygon", [])
    
    point_idx = 0
    for field in fields_order:
        # If field exists (is not null), we take another two points from polygon
        if chapter.get(field) is not None:
            if point_idx + 1 < len(polygon):
                p1 = tuple(polygon[point_idx])
                p2 = tuple(polygon[point_idx + 1])
                
                # Drawing rectangle
                color = COLORS.get(field, (255, 255, 255))
                cv2.rectangle(img, p1, p2, color, 2)
                
                # Optionally adding label with field name
                # cv2.putText(img, field, (p1[0], p1[1] - 5), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                point_idx += 2
    
    # Subchapter recursion
    for sub in chapter.get("subchapters", []):
        draw_chapter(img, sub)

def main():
   # Parse CLI arguments
    args = parse_arguments()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created folder: {args.output_dir}")

    # Find all JSON files using the path from arguments
    json_files = glob.glob(os.path.join(args.json_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {args.json_dir}")
        return

    for json_path in json_files:
        print(f"Processing: {os.path.basename(json_path)}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Getting image name - same name as JSON expected, but looking in args.image_dir
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = os.path.join(args.image_dir, base_name + ".jpg")
        
        if not os.path.exists(img_path):
            print(f" WARNING: Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f" FAIL: Could not read image {img_path}")
            continue

        # Go through all the main entities in JSON
        for entry in data:
            draw_chapter(img, entry)
            
        # Saving output to the path from arguments
        out_path = os.path.join(args.output_dir, base_name + "_vis.jpg")
        cv2.imwrite(out_path, img)
        print(f" Saved to: {out_path}")

if __name__ == "__main__":
    main()
