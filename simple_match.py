import logging
from detector_parser import DetectorParser
from pero_ocr import ALTOMatch
import json
import os


##################################################
# just for a check whether the matching even works
##################################################



# logging.basicConfig(level=logging.INFO)           # logging
logging.basicConfig(level=logging.CRITICAL + 1)     # quiet mode

# load json with anotations
export_path = "data/project-38-at-2026-03-04-11-19-c8d8673e.json"                # hardcoded
parser = DetectorParser()
parser.parse_label_studio(export_path, run_checks=False)

# load OCR .xml data
složka_s_xml = "data/digilinka_obsahy.alto"                                      # hardcoded

# match json with xmls
matcher = ALTOMatch(detector_parser=parser, alto_export_dir=složka_s_xml)
print("Matching text with annotations...")
matcher.match()

out_dir = "simple_match_jsons"
os.makedirs(out_dir, exist_ok=True)

print("creating page files")

for page in matcher.matched_pages:

    # Access page data - AnnotatedPage instance
    original_page = page.detector_parser_page 
    
    # dictionary for page data
    page_data = {
        "image_filename": original_page.image_filename,
        "detections": []
    }
    
    # fill it with matched detections
    if page.matched_detections:
        for detection in page.matched_detections:
            bbox = detection.detector_parser_annotated_bounding_box
            
            det_data = {
                "class": detection.get_class(),
                "text": detection.get_text(),
                "bbox": {
                    "id": bbox.id,
                    "x": bbox.x,
                    "y": bbox.y,
                    "width": bbox.width,
                    "height": bbox.height
                }
            }
            page_data["detections"].append(det_data)
            

    if original_page.image_filename:
        filename = os.path.splitext(original_page.image_filename)[0] + ".json"
    else:
        # Just in case image doesnt have a name (prob safe to delete)
        filename = f"{original_page.id}.json"
        
    filepath = os.path.join(out_dir, filename)
    
    # save
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(page_data, f, ensure_ascii=False, indent=4)

print(f"Done, {len(matcher.matched_pages)} files saved to '{out_dir}'.")