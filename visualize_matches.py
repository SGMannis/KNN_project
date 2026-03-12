import argparse
import cv2
import json
import os
import glob

# Default Path configuration
DEFAULT_JSON_DIR = "out/"
DEFAULT_IMAGE_DIR = "data/project-38-at-2026-02-24-13-47-846604c7/images/"
DEFAULT_OUTPUT_DIR = "out_vis/"

# Color definition for separate elements (BGR format)
COLORS = {
    "name": (0, 0, 255),           # red
    "chapter_number": (255, 0, 0), # blue
    "page_number": (0, 255, 0),    # green
    "description": (0, 255, 255),  # yellow
    "line": (0, 0, 0)
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

def draw_chapter(img, chapter):
    """Recursively draws chapter bboxes and connects their centers with a line to show alignment."""
    fields = ["name", "chapter_number", "page_number", "description"]
    centers = []
    
    for field in fields:
        bbox_key = f"{field}_bbox"
        bbox = chapter.get(bbox_key)
        
        # Draw only if the bbox exists and is not null
        if bbox and len(bbox) == 2:
            # Ensure coordinates are tuples of integers
            p1 = tuple(map(int, bbox[0]))
            p2 = tuple(map(int, bbox[1]))
            
            # Calculate the center point for the connecting line
            cx = (p1[0] + p2[0]) // 2
            cy = (p1[1] + p2[1]) // 2
            centers.append((cx, cy))
            
            color = COLORS.get(field, (255, 255, 255))
            
            # Draw the bounding box
            cv2.rectangle(img, p1, p2, color, 2)
            
    # Draw connecting lines between the centers to show horizontal alignment
    if len(centers) > 1:
        # Sort points by X-coordinate so the line follows the reading direction
        centers.sort(key=lambda p: p[0])
        for i in range(len(centers) - 1):
            # Using a light gray color (180, 180, 180) for the connecting line
            cv2.line(img, centers[i], centers[i+1], COLORS["line"], 2, cv2.LINE_AA)
    
    # Subchapter recursion
    for sub in chapter.get("subchapters", []):
        draw_chapter(img, sub)

def main():
    args = parse_arguments()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created folder: {args.output_dir}")

    # Find all JSON files
    json_files = glob.glob(os.path.join(args.json_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {args.json_dir}")
        return

    for json_path in json_files:
        filename = os.path.basename(json_path)
        print(f"Processing: {filename}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f" FAIL: Could not read JSON {json_path}: {e}")
            continue
        
        # Image name matching logic
        base_name = os.path.splitext(filename)[0]
        img_path = os.path.join(args.image_dir, base_name + ".jpg")
        
        if not os.path.exists(img_path):
            print(f" WARNING: Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f" FAIL: OpenCV could not read image {img_path}")
            continue

        # Process each top-level chapter object in the list
        for entry in data:
            draw_chapter(img, entry)
            
        # Save visualization
        out_path = os.path.join(args.output_dir, base_name + "_vis.jpg")
        cv2.imwrite(out_path, img)
        print(f" Saved to: {out_path}")

if __name__ == "__main__":
    main()
