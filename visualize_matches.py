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
    "name": (0, 0, 255),           # Červená
    "chapter_number": (255, 0, 0), # Modrá
    "page_number": (0, 255, 0),    # Zelená
    "description": (0, 255, 255)   # Žlutá
}

def draw_chapter(img, chapter):
    """Rekurzivně vykreslí kapitolu a její podkapitoly."""
    fields_order = ["name", "chapter_number", "page_number", "description"]
    polygon = chapter.get("polygon", [])
    
    point_idx = 0
    for field in fields_order:
        # If field exists (is not null), we take another two point from polygon
        if chapter.get(field) is not None:
            if point_idx + 1 < len(polygon):
                p1 = tuple(polygon[point_idx])
                p2 = tuple(polygon[point_idx + 1])
                
                # Drawing rectangle
                color = COLORS.get(field, (255, 255, 255))
                cv2.rectangle(img, p1, p2, color, 2)
                
                # Optionally adding label with field name
                cv2.putText(img, field, (p1[0], p1[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                point_idx += 2
    
    # Subchapter recursion
    for sub in chapter.get("subchapters", []):
        draw_chapter(img, sub)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created folder: {OUTPUT_DIR}")

    # Find all JSON files in JSON_DIR folder
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))
    
    if not json_files:
        print("No JSON files found in", JSON_DIR)
        return

    for json_path in json_files:
        print(f"Processing: {os.path.basename(json_path)}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Getting image name - same name as JSON expected, except with .jpg extension
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = os.path.join(IMAGE_DIR, base_name + ".jpg")
        
        if not os.path.exists(img_path):
            print(f" WARNING: Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f" FAIL: Could not read image {img_path}")
            continue

        # Go though all the main entities in JSON
        for entry in data:
            draw_chapter(img, entry)
            
        # Saving output
        out_path = os.path.join(OUTPUT_DIR, base_name + "_vis.jpg")
        cv2.imwrite(out_path, img)
        print(f" Saved to: {out_path}")

if __name__ == "__main__":
    main()
