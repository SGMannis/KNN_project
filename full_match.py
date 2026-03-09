import logging
# Import your two classes
from detector_parser import DetectorParser
from pero_ocr import ALTOMatch
import json
import os
import argparse

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
    # STEP 2: Creating anchors (Chapters)
    # ==========================================
    for chap in chapters:
        bbox_chap = chap.detector_parser_annotated_bounding_box 
        y_chap_top = bbox_chap.y
        y_chap_center = bbox_chap.y + (bbox_chap.height / 2) 
        y_chap_bottom = bbox_chap.y + bbox_chap.height 
        
        group = {
            "record_type": "1",
            "title_detection": chap,
            "page_number_detection": None,
            "chapter_number_detection": None,
            "subheading": None,
            "items": [], 
            "_y_top": y_chap_top,
            "_y_center": y_chap_center,
            "_y_bottom": y_chap_bottom,
            "_x": bbox_chap.x, 
            "_id": bbox_chap.id 
        }

        # Pairing the page number for the chapter (based on relation annotation)
        id_chap = group["_id"]
        if id_chap in relation_to_from and relation_to_from[id_chap] in detection_by_id:
            target_detection = detection_by_id[relation_to_from[id_chap]]
            if target_detection.get_class() == "cislo strany": 
                group["page_number_detection"] = target_detection
                # remove from list
                if target_detection in page_numbers:
                    page_numbers.remove(target_detection)

        # Pairing the page number for the chapter (based on Geometry)
        # TODO maybe delete and rely only on annotation
        if group["page_number_detection"] is None:
            for page in page_numbers:
                bbox_page = page.detector_parser_annotated_bounding_box 
                y_page = bbox_page.y + bbox_page.height
                # chapter name and its page number (bottoms) are in approx. same height and chapter page number is to the right of the chapter name
                if abs(group["_y_bottom"] - y_page) <= n_of_pixels_spare and bbox_page.x > group["_x"]: 
                    group["page_number_detection"] = page
                    # remove from list
                    page_numbers.remove(page)
                    break
    
        # Pairing the chapter number
        for c_chap in chapter_numbers:
            bbox_c_chap = c_chap.detector_parser_annotated_bounding_box 
            # chapter name and its number are in approx. same height and chapter number is to the left of the chapter name
            if abs(group["_y_top"] - bbox_c_chap.y) <= n_of_pixels_spare and bbox_c_chap.x < group["_x"]: 
                group["chapter_number_detection"] = c_chap
                # remove from list
                chapter_numbers.remove(c_chap)
                break
        
        resulting_groups.append(group)

    
    orphans = {
        "record_type": "N/A",
        "title_detection": None, # Orphans don't have a main heading detection
        "page_number_detection": None,
        "chapter_number_detection": None,
        "items": [],
        "_y_center": -1000 
    }

    def find_parent_chapter(y_item):
        closest = None
        min_distance = float('inf')
        for group in resulting_groups:
            if y_item > group["_y_center"]: 
                distance = y_item - group["_y_center"]
                if distance < min_distance:
                    min_distance = distance
                    closest = group
        return closest if closest is not None else orphans
    

    # ==========================================
    # STEP 3: Other headings (sometimes numbered and can have a page number)
    # ==========================================
    for o_heading in other_headings:
        bbox_o = o_heading.detector_parser_annotated_bounding_box 
        y_o_heading_top = bbox_o.y
        y_o_heading_center = bbox_o.y + (bbox_o.height / 2) 
        y_o_heading_bottom = bbox_o.y + bbox_o.height
        id_o = bbox_o.id 
        
        parent = find_parent_chapter(y_o_heading_center)
        page_detection = None
        chap_num_detection = None
        
        # Looking for the page number (Relation -> Geometry)
        if id_o in relation_to_from and relation_to_from[id_o] in detection_by_id:
            target_detection = detection_by_id[relation_to_from[id_o]]
            if target_detection.get_class() == "cislo strany": 
                page_detection = target_detection
                # remove from list
                if target_detection in page_numbers:
                    page_numbers.remove(target_detection)
                
        if page_detection is None:
            for page in page_numbers:
                bbox_page = page.detector_parser_annotated_bounding_box 
                y_page = bbox_page.y + bbox_page.height
                # subheading and its page number (bottoms) are in approx. same height and page number is to the right of the subheading
                if abs(y_o_heading_bottom - y_page) <= n_of_pixels_spare and bbox_page.x > bbox_o.x: 
                    page_detection = page
                    # remove from list
                    page_numbers.remove(page)
                    break

        for c_chap in chapter_numbers:
            bbox_c = c_chap.detector_parser_annotated_bounding_box
            # subheading and its number are in approx. same height and subheading number is to the left of the subheading
            if abs(y_o_heading_top - bbox_c.y) <= n_of_pixels_spare and bbox_c.x < bbox_o.x:
                chap_num_detection = c_chap
                # remove from list
                chapter_numbers.remove(c_chap)
                break
                    
        parent["items"].append({
            "record_type": "4",
            "detection": o_heading,
            "page_detection": page_detection,
            "number_detection": chap_num_detection,
            "_y_center": y_o_heading_top
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
        group["items"].sort(key=lambda p: p["_y_center"])
        
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



if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Annotations, OCR output and matched output")

    argparser.add_argument(
        "-j", "--json_annotations", 
        type=str,
        default="data/project-38-at-2026-03-04-11-19-c8d8673e.json",
        help="path to the exported json annotations file"
    )

    argparser.add_argument(
        "-c", "--ocr_dir",
        type=str,
        default="data/digilinka_obsahy.alto", 
        help="path to the OCR output directory"
    )

    argparser.add_argument(
        "-o", "--output_dir", 
        type=str,
        default="out/",
        help="path to the output directory"
    )

    args = argparser.parse_args()
        
    # logging.basicConfig(level=logging.INFO)           # logging
    logging.basicConfig(level=logging.CRITICAL + 1)     # quiet mode

    # load json with annotations
    export_path = args.json_annotations
    parser = DetectorParser()
    parser.parse_label_studio(export_path, run_checks=False)

    # load OCR .xml data
    ocr_dir = args.ocr_dir

    # match json with xmls
    matcher = ALTOMatch(detector_parser=parser, alto_export_dir=ocr_dir)
    print("Matching text with annotations...")
    matcher.match()

    # output directory
    output_dir = args.output_dir
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
