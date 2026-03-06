import logging
# Import your two classes
from detector_parser import DetectorParser
from pero_ocr import ALTOMatch
import json
import os


##################################################
# Match OCR and annotations and create dataset
##################################################



def group_items_on_page(matched_page):
    # STEP 0: Preparation - map bounding box IDs to extracted data for quick lookup
    detection_by_id = {}
    for detection in matched_page.matched_detections:
        bbox = detection.detector_parser_annotated_bounding_box
        detection_by_id[bbox.id] = detection

    n_of_pixels_spare = 15

    # Access page data - AnnotatedPage instance
    original_page = matched_page.detector_parser_page 

    relation_to_from = {rel.to_id: rel.from_id for rel in original_page.relations}

    chapters = []
    other_headings = []
    subheadings = []
    chapter_numbers = []
    page_numbers = []

    # ==========================================
    # STEP 1: Sorting into buckets and filtering
    # ==========================================
    for detection in matched_page.matched_detections:
        cls = detection.get_class()
        
        if cls == "kapitola":
            chapters.append(detection)
        elif cls == "jiny nadpis":
            other_headings.append(detection)
        elif cls == "podnadpis":
            subheadings.append(detection)
        elif cls == "jine cislo":
            chapter_numbers.append(detection)
        elif cls == "cislo strany":
            page_numbers.append(detection)

    chapters.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)
    page_numbers.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)
    chapter_numbers.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)
    other_headings.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)
    subheadings.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)
    
    resulting_groups = []

    # ==========================================
    # STEP 2: Creating anchors (Chapters - "1")
    # ==========================================
    for chap in chapters:
        bbox_chap = chap.detector_parser_annotated_bounding_box 
        y_chap = bbox_chap.y + (bbox_chap.height / 2) 
        
        group = {
            "record_type": "1",
            "title_detection": chap,
            "page_number_detection": None,
            "chapter_number_detection": None,
            "subheading": None,
            "items": [], 
            "_y": y_chap,
            "_x": bbox_chap.x, 
            "_id": bbox_chap.id 
        }

        # Pairing the page number for the chapter (based on relation annotation)
        id_chap = group["_id"]
        if id_chap in relation_to_from and relation_to_from[id_chap] in detection_by_id:
            target_detection = detection_by_id[relation_to_from[id_chap]]
            if target_detection.get_class() == "cislo strany": 
                group["page_number_detection"] = target_detection

        # Pairing the page number for the chapter (based on Geometry)
        if group["page_number_detection"] is None: # won't work if it's too far away
            for page in page_numbers:
                bbox_page = page.detector_parser_annotated_bounding_box 
                y_page = bbox_page.y + (bbox_page.height / 2)
                if abs(y_chap - y_page) <= n_of_pixels_spare and bbox_page.x > group["_x"]: 
                    group["page_number_detection"] = page
                    break
    
        # Pairing the chapter number
        # TODO it might be better to match by corners rather than centers
        for c_chap in chapter_numbers:
            bbox_c = c_chap.detector_parser_annotated_bounding_box 
            y_c = bbox_c.y + (bbox_c.height / 2) 
            if abs(y_chap - y_c) <= n_of_pixels_spare and bbox_c.x < group["_x"]: 
                group["chapter_number_detection"] = c_chap
                break
        
        resulting_groups.append(group)

    
    orphans = {
        "record_type": "N/A",
        "title_detection": None, # Orphans don't have a main heading detection
        "page_number_detection": None,
        "chapter_number_detection": None,
        "items": [],
        "_y": -1000 
    }

    def find_parent_chapter(y_item):
        closest = None
        min_distance = float('inf')
        for group in resulting_groups:
            if y_item > group["_y"]: 
                distance = y_item - group["_y"]
                if distance < min_distance:
                    min_distance = distance
                    closest = group
        return closest if closest is not None else orphans
    

    # ==========================================
    # STEP 3: Other headings (sometimes numbered and can have a page number)
    # ==========================================
    for o_heading in other_headings:
        bbox_o = o_heading.detector_parser_annotated_bounding_box 
        y_o = bbox_o.y + (bbox_o.height / 2) 
        id_o = bbox_o.id 
        
        parent = find_parent_chapter(y_o)
        page_detection = None
        chap_num_detection = None
        
        # Looking for the page number (Relation -> Geometry)
        if id_o in relation_to_from and relation_to_from[id_o] in detection_by_id:
            target_detection = detection_by_id[relation_to_from[id_o]]
            if target_detection.get_class() == "cislo strany": 
                page_detection = target_detection
                
        if page_detection is None:
            for page in page_numbers:
                bbox_page = page.detector_parser_annotated_bounding_box 
                y_page = bbox_page.y + (bbox_page.height / 2) 
                if abs(y_o - y_page) <= n_of_pixels_spare and bbox_page.x > bbox_o.x: 
                    page_detection = page
                    break

        # TODO it might be better to match by corners rather than centers
        for c_chap in chapter_numbers:
            bbox_c = c_chap.detector_parser_annotated_bounding_box
            y_c = bbox_c.y + (bbox_c.height / 2)
            if abs(y_o - y_c) <= n_of_pixels_spare and bbox_c.x < bbox_o.x:
                chap_num_detection = c_chap
                break
                    
        parent["items"].append({
            "record_type": "4",
            "detection": o_heading,             # <--- Entire object
            "page_detection": page_detection,   # <--- Entire object
            "number_detection": chap_num_detection, # <--- Entire object
            "_y": y_o
        })

    # ==========================================
    # Subheadings
    # ==========================================
    for subheading in subheadings:
        bbox_sub = subheading.detector_parser_annotated_bounding_box 
        y_sub = bbox_sub.y + (bbox_sub.height / 2) 
        
        parent = find_parent_chapter(y_sub)

        if parent is not orphans:
            parent["subheading"] = subheading

    # ==========================================
    # Sorting (and cleanup) of items inside chapters
    # ==========================================
    if orphans["items"]: 
        resulting_groups.insert(0, orphans)

    for group in resulting_groups:
        group["items"].sort(key=lambda p: p["_y"])
        
        # group.pop("_y", None)
        # group.pop("_x", None)
        # group.pop("_id", None)
        # for p in group["items"]:
        #     p.pop("_y", None)

    return resulting_groups


def process_detection(detection):
    # If we found nothing (e.g., missing page number), return None
    if detection is None:
        return None
        
    # Grab the bounding box
    # bbox = detection.detector_parser_annotated_bounding_box
    
    return {
        "class": detection.get_class(), 
        "text": detection.get_text(), 
        # "bbox": {
        #     "x": bbox.x, 
        #     "y": bbox.y, 
        #     "width": bbox.width, 
        #     "height": bbox.height 
        # }
    }


# logging.basicConfig(level=logging.INFO)           # logging
logging.basicConfig(level=logging.CRITICAL + 1)     # quiet mode

# load json with annotations
export_path = "data/project-38-at-2026-03-04-11-19-c8d8673e.json"               # hardcoded
parser = DetectorParser()
parser.parse_label_studio(export_path, run_checks=False)

# load OCR .xml data
xml_dir = "data/digilinka_obsahy.alto"                                          # hardcoded

# match json with xmls
matcher = ALTOMatch(detector_parser=parser, alto_export_dir=xml_dir)
print("Matching text with annotations...")
matcher.match()

output_dir = "full_match_jsons"
os.makedirs(output_dir, exist_ok=True)

print("Creating files")
# Iterate through all matched pages
for page in matcher.matched_pages:
    # Call your super function
    rich_groups = group_items_on_page(page)
    
    # Prepare clean data for JSON here
    clean_page_data = []
    
    for group in rich_groups:
        # A) Translate the main chapter objects
        clean_group = {
            # "record_type": group.get("record_type"),
            "title": process_detection(group.get("title_detection")),
            "page_number": process_detection(group.get("page_number_detection")),
            "chapter_number": process_detection(group.get("chapter_number_detection")),
            "subheading": process_detection(group.get("subheading")),
            "items": []
        }
        
        # B) Translate nested items (your "other headings")
        for item in group.get("items", []):
            clean_item = {
                # "record_type": item.get("record_type"),
                "title": process_detection(item.get("detection")),
                "page_number": process_detection(item.get("page_detection")),
                "chapter_number": process_detection(item.get("number_detection"))
            }
            clean_group["items"].append(clean_item)
            
        clean_page_data.append(clean_group)
        
    # 3. SAVE THE FILE
    original_page = page.detector_parser_page 
    
    # Generate filename based on the image (e.g., document_01.json)
    if original_page.image_filename:
        file_name = os.path.splitext(original_page.image_filename)[0] + ".json"
    else:
        file_name = f"{original_page.id}.json" 
        
    file_path = os.path.join(output_dir, file_name)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(clean_page_data, f, ensure_ascii=False, indent=4)
        
print(f"Done")
