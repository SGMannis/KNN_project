import logging
# Import your two classes
from detector_parser import DetectorParser
from pero_ocr import ALTOMatch
import json
import os
import argparse
from pydantic import BaseModel
from typing import Optional, List, Tuple

##################################################
# Match OCR and annotations and create dataset
##################################################

class Chapter(BaseModel):
    name: Optional[str] = None
    chapter_number: Optional[str] = None
    page_number: Optional[str] = None
    description: Optional[str] = None

    name_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    chapter_number_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    page_number_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    description_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

    name_conf: Optional[List[float]] = []
    chapter_number_conf: Optional[List[float]] = []
    page_number_conf: Optional[List[float]] = []
    description_conf: Optional[List[float]] = []

    subchapters: Optional[List['Chapter']] = []



def get_corner_points(detection) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if not detection:
        return None
        
    bbox = detection.detector_parser_annotated_bounding_box
    x, y = int(bbox.x), int(bbox.y)
    w, h = int(bbox.width), int(bbox.height)
    
    # Top-Left and Bottom-Right 
    return ((x , y), (x + w, y + h))



def get_text_safe(detection) -> Optional[str]:
    if not detection:
        return None
    text = detection.get_text().strip()    
    return text if text != "" else None



def get_confs_safe(detection) -> Optional[List[float]]:
    return detection.get_word_confidences() if detection else None



def group_items_on_page(matched_page):
    # ==========================================
    # Preparation - map bounding box IDs to extracted data for quick lookup and computing page width
    # ==========================================
    detection_by_id = {}
    all_x_min = []
    all_x_max = []

    # annotated relations
    relation_to_from = {rel.to_id: rel.from_id for rel in original_page.relations}
    
    # setup detection by id and prep min/max coords of the page
    for detection in matched_page.matched_detections:
        bbox = detection.detector_parser_annotated_bounding_box
        detection_by_id[bbox.id] = detection
        all_x_min.append(bbox.x)
        all_x_max.append(bbox.x + bbox.width)

    page_min_x = min(all_x_min) if all_x_min else 0
    page_max_x = max(all_x_max) if all_x_max else 1000
    page_width = page_max_x - page_min_x
    mid_x = page_min_x + (page_width / 2)

    gutter_width = page_width * 0.05    
    n_of_pixels_spare = 15

    # Access page data - AnnotatedPage instance
    original_page = matched_page.detector_parser_page 


    # ==========================================
    # Sorting into buckets and filtering + column tags
    # ==========================================

    # in case of twocolumn layout - any element can be tagged left, full or right
    column_tags = {} 
    
    chapters = []
    other_headings = []
    subheadings = []
    chapter_numbers = []
    page_numbers = []

    twocolumn = False
    for detection in matched_page.matched_detections:
        cls = detection.get_class()
        bbox = detection.detector_parser_annotated_bounding_box
        
        if cls == "kapitola":
            chapters.append(detection)
        elif cls == "jiny nadpis":
            other_headings.append(detection)
        elif cls == "podnadpis":
            subheadings.append(detection)
        elif cls == "jine cislo":
            chapter_numbers.append(detection)
            continue
        elif cls == "cislo strany":
            page_numbers.append(detection)
            continue
        
        # bbox starts on the left and ends on the right side
        if bbox.x < (mid_x - gutter_width) and (bbox.x + bbox.width) > (mid_x + gutter_width):
            column_tags[bbox.id] = "full"
        # bbox starts on the right side
        elif bbox.x > mid_x:
            column_tags[bbox.id] = "right"
            twocolumn = True
        else:
            column_tags[bbox.id] = "left"
    

    # sort descending
    chapters.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)
    page_numbers.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)
    chapter_numbers.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)
    other_headings.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)
    subheadings.sort(key=lambda d: d.detector_parser_annotated_bounding_box.y)

    all_headings = chapters + other_headings


    # ==========================================
    # Helper functions checking obstacles between possibly linked detections
    # ==========================================

    # start_x - original heading, dest_x - potential (sub)chapter number (x coords), y_line - y level of the heading
    def left_path_free(start_x, dest_x, y_line):
        for h in all_headings:
            h_bbox = h.detector_parser_annotated_bounding_box
            h_bbox_center = h_bbox.x + (h_bbox.width / 2)
            # collision based on x axis? Obstacle right of (sub)chapter number and left of original heading
            if  dest_x < h_bbox_center and h_bbox_center < start_x:
                # same line?
                if (h_bbox.y - 5) <= y_line <= (h_bbox.y + h_bbox.height + 5):
                    return False # we are in a wrong column
        return True # ok
    
    # start_x - original heading, dest_x - potential page number (x coords), y_line - y level of the heading
    def right_path_free(start_x, dest_x, y_line):
        for h in all_headings:
            h_bbox = h.detector_parser_annotated_bounding_box
            h_bbox_center = h_bbox.x + (h_bbox.width / 2)
            # collision based on x axis? Obstacle right of original heading and left of page number
            if start_x < h_bbox_center and h_bbox_center < dest_x:
                # same line?
                if (h_bbox.y - 5) <= y_line <= (h_bbox.y + h_bbox.height + 5):
                    return False # we are in a wrong column
        return True # ok
    
    # stary_y - original heading, dest_y - potential chapter number (y coords), , x_line - x level of the heading
    def up_path_free(start_y, dest_y, x_line):
        for h in matched_page.matched_detections:
            h_bbox = h.detector_parser_annotated_bounding_box
            h_bbox_center_y = h_bbox.y + (h_bbox.height / 2)
            
            if dest_y < h_bbox_center_y < start_y:                
                if (h_bbox.x - 5) <= x_line <= (h_bbox.x + h_bbox.width + 5):
                    return False # Narazili jsme na překážku nad námi
                    
        return True # ok
    
    
    resulting_groups = []

    # ==========================================
    # Chapters
    # ==========================================
    for chap in chapters:
        bbox_chap = chap.detector_parser_annotated_bounding_box 
        y_chap_top = bbox_chap.y
        y_chap_center = bbox_chap.y + (bbox_chap.height / 2) 
        y_chap_bottom = bbox_chap.y + bbox_chap.height 
        x_chap_center = bbox_chap.x + (bbox_chap.width / 2)
        
        group = {
            "title_detection": chap,
            "chapter_number_detection": None,
            "page_number_detection": None,
            "subheading_detection": None,
            "items": [], 
            "_y_top": y_chap_top,
            "_y_center": y_chap_center,
            "_y_bottom": y_chap_bottom,
            "_x": bbox_chap.x,
            "_x_center": x_chap_center,
            "_id": bbox_chap.id,
            "_tag": column_tags[bbox_chap.id]
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
        if group["page_number_detection"] is None: # only when relation match fails (should succeed every time but ehhhh)
            for page in page_numbers:
                bbox_page = page.detector_parser_annotated_bounding_box 
                y_page = bbox_page.y + bbox_page.height
                # chapter name and its page number (bottoms) are in approx. same height and chapter page number is to the right of the chapter name
                if abs(group["_y_bottom"] - y_page) <= n_of_pixels_spare and bbox_page.x > group["_x"]: 
                    if right_path_free(group["_x"], bbox_page.x, group["_y_top"]):
                        group["page_number_detection"] = page
                        # remove from list
                        page_numbers.remove(page)
                        break
    
        # Pairing the chapter number
        for c_chap in chapter_numbers:
            bbox_c_chap = c_chap.detector_parser_annotated_bounding_box 
            c_chap_y_center = bbox_c_chap.y + (bbox_c_chap.height / 2)
            if (abs(group["_y_top"] - bbox_c_chap.y) <= n_of_pixels_spare or abs(group["_y_center"] - c_chap_y_center) <= n_of_pixels_spare) and bbox_c_chap.x < group["_x"]: 
                if left_path_free(group["_x"], bbox_c_chap.x, group["_y_top"]):
                    group["chapter_number_detection"] = c_chap
                    # remove from list
                    chapter_numbers.remove(c_chap)
                    break
            # chapter number above chapter
            c_chap_x_center = bbox_c_chap.x + (bbox_c_chap.width / 2)
            if abs(group["_x_center"] - c_chap_x_center) <= n_of_pixels_spare and bbox_c_chap.y < group["_y_center"]:
                if up_path_free(group["_y_center"], c_chap_y_center, group["_x_center"]): # directly above
                    group["chapter_number_detection"] = c_chap
                    # remove from list
                    chapter_numbers.remove(c_chap)
                    break
        
        resulting_groups.append(group)

    
    orphans = {
        "title_detection": None, # Orphans don't have a main heading detection
        "chapter_number_detection": None,
        "page_number_detection": None,
        "subheading_detection": None,
        "items": [],
        "_y_center": -1000,
        "_tag": "full"
    }

    # find chapter for subchapter. Accounts for twocolumn
    def find_parent_chapter(y_item, item_tag):
        closest_above = None
        min_distance = float('inf')
        
        for group in resulting_groups:
            parent_tag = group["_tag"]
            #      same column     || chapter can be in the middle ||  some subchapters in unicolumn are labeled "full", while chapter can be labeled "left" 
            if item_tag == parent_tag or parent_tag == "full" or (item_tag == "full" and parent_tag == "left"):
                if y_item > group["_y_center"]: 
                    distance = y_item - group["_y_center"]
                    if distance < min_distance:
                        min_distance = distance
                        closest_above = group
                        
        if closest_above is not None:
            return closest_above


        if item_tag == "right":
            lowest_left_chapter = None
            max_y = -float('inf')
            
            for group in resulting_groups:
                if group["_tag"] == "left":
                    if group["_y_center"] > max_y:
                        max_y = group["_y_center"]
                        lowest_left_chapter = group
                            
            if lowest_left_chapter is not None:
                return lowest_left_chapter

        return orphans
    

    # ==========================================
    # Other headings (sometimes numbered and can have a page number - rarely a subheading)
    # ==========================================
    for o_heading in other_headings:
        bbox_o = o_heading.detector_parser_annotated_bounding_box 
        y_o_heading_top = bbox_o.y
        y_o_heading_center = bbox_o.y + (bbox_o.height / 2) 
        y_o_heading_bottom = bbox_o.y + bbox_o.height
        id_o = bbox_o.id
        my_tag = column_tags[id_o]
        
        parent = find_parent_chapter(y_o_heading_center, my_tag)
        page_detection = None
        chap_num_detection = None
        
        # Pairing the page number for the subchapter (based on relation annotation)
        if id_o in relation_to_from and relation_to_from[id_o] in detection_by_id:
            target_detection = detection_by_id[relation_to_from[id_o]]
            if target_detection.get_class() == "cislo strany": 
                page_detection = target_detection
                # remove from list
                if target_detection in page_numbers:
                    page_numbers.remove(target_detection)
        
        # # Pairing the page number for the subchapter (based on Geometry)
        if page_detection is None:
            for page in page_numbers:
                bbox_page = page.detector_parser_annotated_bounding_box 
                y_page = bbox_page.y + bbox_page.height
                # subchapter and its page number (bottoms) are in approx. same height and page number is to the right of the subchapter
                if abs(y_o_heading_bottom - y_page) <= n_of_pixels_spare and bbox_page.x > bbox_o.x: 
                    if right_path_free(bbox_o.x, bbox_page.x, y_o_heading_center):
                        page_detection = page
                        # remove from list
                        page_numbers.remove(page)
                        break

        # Pairing the subchapter number
        for c_chap in chapter_numbers:
            bbox_c = c_chap.detector_parser_annotated_bounding_box
            c_chap_y_center = bbox_c.y + (bbox_c.height / 2)
            # subchapter and its number are in approx. same height and subchapter number is to the left of the subchapter
            if (abs(y_o_heading_top - bbox_c.y) <= n_of_pixels_spare or abs(y_o_heading_center - c_chap_y_center) <= n_of_pixels_spare) and bbox_c.x < bbox_o.x:
                if left_path_free(bbox_o.x, bbox_c.x, y_o_heading_center):
                    chap_num_detection = c_chap
                    # remove from list
                    chapter_numbers.remove(c_chap)
                    break
                    
        parent["items"].append({
            "title_detection": o_heading,
            "chapter_number_detection": chap_num_detection,
            "page_number_detection": page_detection,
            "subheading_detection": None,
            "_y_center": y_o_heading_top,
            "_tag": my_tag
        })

    # ==========================================
    # Subheadings
    # ==========================================

    for subheading in subheadings:
        bbox_sub = subheading.detector_parser_annotated_bounding_box 
        y_sub = bbox_sub.y + (bbox_sub.height / 2)
        sub_tag = column_tags[bbox_sub.id]
        
        parent = find_parent_chapter(y_sub, column_tags[bbox_sub.id])
        if parent["_y_center"] < y_sub and (parent["_tag"] == sub_tag or parent["_tag"] == "full" or sub_tag == "full"):
            dist_to_parent = y_sub - parent["_y_center"]
            matchable = True
        else: 
            dist_to_parent = float('inf')
            matchable = False

        closest_subchapter = None
        min_dist_to_sub = float('inf')

        for item in parent["items"]:
            item_tag = item.get("_tag")
            # column check
            if item_tag == sub_tag or item_tag == "full" or sub_tag == "full":
                if item["_y_center"] < y_sub:
                    dist = y_sub - item["_y_center"]
                    if dist < min_dist_to_sub:
                        min_dist_to_sub = dist
                        closest_subchapter = item

        if closest_subchapter is not None and min_dist_to_sub < dist_to_parent:
            closest_subchapter["subheading_detection"] = subheading
        else:
            if parent is not orphans and matchable:
                parent["subheading_detection"] = subheading

    # ==========================================
    # Sorting (and cleanup) of items inside chapters
    # ==========================================
    if orphans["items"]: 
        resulting_groups.insert(0, orphans)

    
    if twocolumn:
        tag_order = {"full": 0, "left": 1, "right": 2}

        for group in resulting_groups:
            group["items"].sort(key=lambda p: (tag_order.get(p.get("_tag", "full"), 0), p["_y_center"]))

        resulting_groups.sort(key=lambda g: (tag_order.get(g.get("_tag", "full"), 0), g["_y_center"]))

    else:
        for group in resulting_groups:
            group["items"].sort(key=lambda p: p["_y_center"])

        resulting_groups.sort(key=lambda g: g["_y_center"])

    return resulting_groups



def parse_arguments():
    parser = argparse.ArgumentParser(description="Annotations, OCR output and matched output")

    parser.add_argument(
        "-j", "--json_annotations", 
        type=str,
        default="data/project-38-at-2026-03-04-11-19-c8d8673e.json",
        help="path to the exported json annotations file"
    )

    parser.add_argument(
        "-c", "--ocr_dir",
        type=str,
        default="data/digilinka_obsahy.alto", 
        help="path to the OCR output directory"
    )

    parser.add_argument(
        "-o", "--output_dir", 
        type=str,
        default="out/",
        help="path to the output directory"
    )

    return parser.parse_args()



if __name__ == "__main__":

    args = parse_arguments()
        
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
            chapter_det = group.get("title_detection")
            chapter_number_det = group.get("chapter_number_detection")
            page_number_det = group.get("page_number_detection")
            description_det = group.get("subheading_detection")

            chapter = Chapter(
                name=get_text_safe(chapter_det) or "No chapter",
                chapter_number=get_text_safe(chapter_number_det),
                page_number=get_text_safe(page_number_det),
                description=get_text_safe(description_det),

                name_bbox=get_corner_points(chapter_det),
                chapter_number_bbox=get_corner_points(chapter_number_det),
                page_number_bbox=get_corner_points(page_number_det),
                description_bbox=get_corner_points(description_det),

                # name_conf=get_confs_safe(chapter_det),
                # chapter_number_conf=get_confs_safe(chapter_number_det),
                # page_number_conf=get_confs_safe(page_number_det),
                # description_conf=get_confs_safe(description_det),

                subchapters=[]
            )

            for item in group.get("items", []):
                item_det = item.get("title_detection")
                chapter_number_det = item.get("chapter_number_detection")
                page_number_det = item.get("page_number_detection")
                subheading_det = item.get("subheading_detection")
                
                subchapter = Chapter(
                    name=get_text_safe(item_det),
                    chapter_number=get_text_safe(chapter_number_det),
                    page_number=get_text_safe(page_number_det),
                    description=get_text_safe(subheading_det),

                    name_bbox=get_corner_points(item_det),
                    chapter_number_bbox=get_corner_points(chapter_number_det),
                    page_number_bbox=get_corner_points(page_number_det),
                    description_bbox=get_corner_points(subheading_det),

                    # name_conf=get_confs_safe(item_det),
                    # chapter_number_conf=get_confs_safe(chapter_number_det),
                    # page_number_conf=get_confs_safe(page_number_det),
                    # description_conf=get_confs_safe(subheading_det)
                )
                chapter.subchapters.append(subchapter)

            clean_page_data.append(chapter.model_dump())

            
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
