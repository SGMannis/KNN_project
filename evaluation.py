import os
import argparse
from pydantic import BaseModel, TypeAdapter, ConfigDict, ValidationError
from typing import Optional, List, Tuple
import jiwer
import torch
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
import difflib
import numpy as np
# import json
# import json_repair


class Chapter(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: Optional[str]
    chapter_number: Optional[str]
    page_number: Optional[str]
    description: Optional[str]

    name_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    chapter_number_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    page_number_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
    description_bbox: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]

    subchapters: Optional[List['Chapter']]



def flatten_bbox(bbox): 
    """ [[x1, y1], [x2, y2]] to [x1, y1, x2, y2] """
    return [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]



def extract_data(chapter_list): # -> (8-tuple)

    # strings might end up with multiple spaces -> take care of that before computing CER
    names = ""
    chapter_numbers = ""
    page_numbers = ""
    descriptions = ""

    name_bboxes = []
    chapter_number_bboxes = []
    page_number_bboxes = []
    description_bboxes = []
    
    for chap in chapter_list:
        ################################
        # texts
        ################################
        if chap.name:
            # names.append(chap.name)
            names = names + " " + chap.name
        
        if chap.chapter_number:
            # chapter_numbers.append(chap.chapter_number)
            chapter_numbers = chapter_numbers + " " + chap.chapter_number

        if chap.page_number:
            # page_numbers.append(chap.page_number)
            page_numbers = page_numbers + " " + chap.page_number

        if chap.description:
            # descriptions.append(chap.description)
            descriptions = descriptions + " " + chap.description

        ################################
        # bboxes
        ###############################
        if chap.name_bbox:
            name_bboxes.append(flatten_bbox(chap.name_bbox))
        
        if chap.chapter_number_bbox:
            chapter_number_bboxes.append(flatten_bbox(chap.chapter_number_bbox))

        if chap.page_number_bbox:
            page_number_bboxes.append(flatten_bbox(chap.page_number_bbox))

        if chap.description_bbox:
            description_bboxes.append(flatten_bbox(chap.description_bbox))

        ################################
        # subchapters
        ###############################
        if chap.subchapters:
            (sub_names, sub_cnum, sub_pnum, sub_desc,
             sub_names_bb, sub_cnum_bb, sub_pnum_bb, sub_desc_bb) = extract_data(chap.subchapters)
            
            # texts
            names = names + " " + sub_names
            chapter_numbers = chapter_numbers + " " + sub_cnum
            page_numbers = page_numbers + " " + sub_pnum
            descriptions = descriptions + " " + sub_desc
            
            # bboxes (lists)
            name_bboxes = name_bboxes + sub_names_bb
            chapter_number_bboxes = chapter_number_bboxes + sub_cnum_bb
            page_number_bboxes = page_number_bboxes + sub_pnum_bb
            description_bboxes = description_bboxes + sub_desc_bb

    return (names, chapter_numbers, page_numbers, descriptions,
            name_bboxes, chapter_number_bboxes, page_number_bboxes, description_bboxes)



def concatenate_data(accumulated, new_data_tuple):
    for i in range(8):
        accumulated[i].append(new_data_tuple[i])



def eval_text(gt_texts, model_texts):
    """ Evaluate texts of all categories """

    transformation_text = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip()
    ])

    transformation_numbers = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation()
    ])

    names_cer = jiwer.cer(
        transformation_text(gt_texts[0]),
        transformation_text(model_texts[0])
    )

    cnums_cer = jiwer.cer(
        transformation_numbers(gt_texts[1]),
        transformation_numbers(model_texts[1])
    )

    pnums_cer = jiwer.cer(
        transformation_numbers(gt_texts[2]),
        transformation_numbers(model_texts[2])
    )

    descs_cer = jiwer.cer(
        transformation_text(gt_texts[3]),
        transformation_text(model_texts[3])
    )

    return (names_cer, cnums_cer, pnums_cer, descs_cer)



def eval_bb_one_file(gt_boxes, pred_boxes, threshold=0.75):
    """ Evaluate bboxes of one category of one file """

    # Edge cases: Handle scenarios where one or both lists are empty
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0, 0.0  # (TP, FP, FN, sum_iou)
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, 0.0 # Everything is FP (hallucinations)
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), 0.0 # Everything is FN (missing data)

    # Convert lists to tensors for box_iou calculation
    gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)
    pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)

    # Calculate IoU matrix between all GT and Predicted boxes
    iou_matrix = box_iou(gt_tensor, pred_tensor).numpy()

    # --- FIX: Clean the matrix ---
    # Replace NaN or Inf values with 0.0. 
    # Invalid entries occur if boxes have zero or negative area (degenerate boxes).
    iou_matrix = np.nan_to_num(iou_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        # Find the optimal assignment (row/column pairs) to maximize total IoU
        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
    except ValueError as e:
        # Fallback if the matrix is still problematic
        print(f" [!] linear_sum_assignment failed: {e}")
        return 0, len(pred_boxes), len(gt_boxes), 0.0

    tp, sum_iou = 0, 0.0

    # Evaluate matches based on the IoU threshold
    for r, c in zip(row_ind, col_ind):
        iou_val = iou_matrix[r, c]
        if iou_val >= threshold:
            tp += 1
            sum_iou += iou_val

    # Final stats calculation
    fp = len(pred_boxes) - tp  # Predicted boxes that didn't match any GT
    fn = len(gt_boxes) - tp    # GT boxes that weren't detected correctly

    return tp, fp, fn, sum_iou



# each element in list represents bboxes from one file (of one category)
def eval_bboxes(gt_bb_list, model_bb_list):
    """ Evaluate bboxes of one category - get f1 score and mean IoU """
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_sum_iou = 0.0

    for gt_page_list, model_page_list in zip(gt_bb_list, model_bb_list):
        tp, fp, fn, sum_iou = eval_bb_one_file(gt_page_list, model_page_list)

        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_sum_iou += sum_iou

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0

    f1_score = 0.0
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
            
    mean_iou = global_sum_iou / global_tp if global_tp > 0 else 0.0

    return f1_score, mean_iou



def calculate_iou_1v1(boxA, boxB):
    """ IoU of two bboxes """
    if not boxA or not boxB:
        return 0.0
        
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    return interArea / float(boxAArea + boxBArea - interArea)


def fuzzy_match(str1, str2):
    """ String similarity (score from 0 to 1) """
    if not str1 and not str2:
        return 1.0  # perfect match
    if not str1 or not str2:
        return 0.0  # one missing - score 0
        
    # delete multiple spaces
    s1_clean = " ".join(str(str1).split())
    s2_clean = " ".join(str(str2).split())
    
    return difflib.SequenceMatcher(None, s1_clean, s2_clean).ratio()


def evaluate_attribute_combined(gt_text, pred_text, gt_bbox, pred_bbox):
    """ Evaluate whether two items (ground truth and model output) are a match. Returns score from 0 to 1 """

    # text similarity
    text_sim = fuzzy_match(gt_text, pred_text)
    
    gt_flat_bbox = flatten_bbox(gt_bbox) if gt_bbox else None
    pred_flat_bbox = flatten_bbox(pred_bbox) if pred_bbox else None
    
    # bbox similarity - IoU
    bbox_sim = calculate_iou_1v1(gt_flat_bbox, pred_flat_bbox)
    
    if gt_bbox or pred_bbox:
        return (text_sim * 0.5) + (bbox_sim * 0.5)
    else:
        return text_sim


def eval_structure(gt_chapters, pred_chapters, threshold=0.75):
    """ Evaluate structure of one file (GT and model output) """
        
    stats = {
        "total_gt_nodes": len(gt_chapters) if gt_chapters else 0,
        "total_pred_nodes": len(pred_chapters) if pred_chapters else 0,
        "matched_nodes": 0,
        "page_number_score": 0.0,
        "chapter_number_score": 0.0,
        "description_score": 0.0,
        "eval_page_numbers": 0,
        "eval_chapter_numbers": 0,
        "eval_descriptions": 0
    }

    if not gt_chapters or not pred_chapters:
        return stats

    # create matrix of similarity based on texts and bboxes of two chapters
    sim_matrix = np.zeros((len(gt_chapters), len(pred_chapters)))
    for i, gt_c in enumerate(gt_chapters):
        for j, pred_c in enumerate(pred_chapters):
            sim_matrix[i, j] = evaluate_attribute_combined(gt_c.name, pred_c.name, gt_c.name_bbox, pred_c.name_bbox)

    # get most likely matches
    row_ind, col_ind = linear_sum_assignment(sim_matrix, maximize=True)

    # evaluate if match
    for r, c in zip(row_ind, col_ind):
        if sim_matrix[r, c] >= threshold: # match or pass
            stats["matched_nodes"] += 1
            
            gt_match = gt_chapters[r]
            pred_match = pred_chapters[c]

            if gt_match.page_number or pred_match.page_number:
                stats["eval_page_numbers"] += 1
                stats["page_number_score"] += evaluate_attribute_combined(gt_match.page_number, pred_match.page_number, gt_match.page_number_bbox, pred_match.page_number_bbox)

            if gt_match.chapter_number or pred_match.chapter_number:
                stats["eval_chapter_numbers"] += 1
                stats["chapter_number_score"] += evaluate_attribute_combined(gt_match.chapter_number, pred_match.chapter_number, gt_match.chapter_number_bbox, pred_match.chapter_number_bbox)
            
            if gt_match.description or pred_match.description:
                stats["eval_descriptions"] += 1
                stats["description_score"] += evaluate_attribute_combined(gt_match.description, pred_match.description, gt_match.description_bbox, pred_match.description_bbox)
                
            # evaluate subchapters
            sub_stats = eval_structure(
                gt_match.subchapters,
                pred_match.subchapters
            )

            # aggregate
            for key in stats:
                stats[key] += sub_stats[key]

    return stats



def parse_arguments():
    parser = argparse.ArgumentParser(description="Annotations, OCR output and matched output")

    parser.add_argument(
        "-g", "--gt_data", 
        type=str,
        default="data_gt/",
        help="dir with ground truth jsons"
    )

    parser.add_argument(
        "-m", "--model_data",
        type=str,
        default="data_model/", 
        help="dir with model output jsons"
    )

    parser.add_argument(
        "-e", "--eval_file", 
        type=str,
        default="eval.txt",
        help="path to the output directory"
    )

    parser.add_argument(
        "-p", "--pretty",
        action='store_true',
        help="enable pretty print. Output is csv if not inserted")

    return parser.parse_args()




if __name__ == "__main__":

    # args
    args = parse_arguments()
    gt_dir = args.gt_data
    model_dir = args.model_data
    outfile = args.eval_file
    pretty = args.pretty

    # dirs
    if not os.path.isdir(gt_dir):
        print(f"Dir '{gt_dir}' not found")
        exit()
    if not os.path.isdir(model_dir):
        print(f"Dir '{model_dir}' not found")
        exit()

    # prepare stats
    valid = 0
    count = 0
    acc_gt = ([],[],[],[],[],[],[],[])
    acc_model = ([],[],[],[],[],[],[],[])

    global_structure_stats = {
        "total_gt_nodes": 0,
        "total_pred_nodes": 0,
        "matched_nodes": 0,
        "page_number_score": 0.0,
        "chapter_number_score": 0.0,
        "description_score": 0.0,
        "eval_page_numbers": 0,
        "eval_chapter_numbers": 0,
        "eval_descriptions": 0
    }

    chapter_list_adapter = TypeAdapter(list[Chapter])

    # go through jsons
    for filename in os.listdir(model_dir):
        if not filename.endswith('.json'):
            continue

        model_filepath = os.path.join(model_dir, filename)
        gt_filepath = os.path.join(gt_dir, filename)

        ########################################################################################
        ########################################################################################
        if not os.path.exists(gt_filepath):
            print(f"skipping {filename} - ground truth not found in {gt_dir}")
            continue

        try:
            with open(gt_filepath, 'r', encoding='utf-8') as f:
                json_content = f.read()
            gt_chapters = chapter_list_adapter.validate_json(json_content)

            with open(model_filepath, 'r', encoding='utf-8') as f:
                json_content = f.read()
            model_chapters = chapter_list_adapter.validate_json(json_content)
            
            valid += 1
            count += 1

        except ValidationError:
            print(f"pydantic validation error {filename} - skipping") # we don't parse invalid jsons
            count += 1
            continue

        except Exception as e:
            print(f"error {e} while processing {filename} - skipping")
            continue
        ########################################################################################

        # extract data from 1 file and concatenate
        file_stats = eval_structure(gt_chapters, model_chapters)
        for key in global_structure_stats:
            global_structure_stats[key] += file_stats[key]

        gt_data = extract_data(gt_chapters)
        model_data = extract_data(model_chapters)

        concatenate_data(acc_gt, gt_data)
        concatenate_data(acc_model, model_data)


    # failsafe
    if count == 0:
        exit()

    # compute stats
    # structure of json
    precision = global_structure_stats["matched_nodes"] / global_structure_stats["total_pred_nodes"] if global_structure_stats["total_pred_nodes"] != 0 else None
    recall = global_structure_stats["matched_nodes"] / global_structure_stats["total_gt_nodes"] if global_structure_stats["total_gt_nodes"] != 0 else None
    struct_f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else None

    page_num_acc = global_structure_stats["page_number_score"] / global_structure_stats["eval_page_numbers"] if global_structure_stats["eval_page_numbers"] != 0 else None
    chap_num_acc = global_structure_stats["chapter_number_score"] / global_structure_stats["eval_chapter_numbers"] if global_structure_stats["eval_chapter_numbers"] != 0 else None
    desc_acc = global_structure_stats["description_score"] / global_structure_stats["eval_descriptions"] if global_structure_stats["eval_descriptions"] != 0 else None

    # text
    (names_cer, cnums_cer, pnums_cer, descs_cer) = eval_text(acc_gt, acc_model)

    # bboxes
    names_bb_f1, names_bb_miou = eval_bboxes(acc_gt[4], acc_model[4])
    cnums_bb_f1, cnums_bb_miou = eval_bboxes(acc_gt[5], acc_model[5])
    pnums_bb_f1, pnums_bb_miou = eval_bboxes(acc_gt[6], acc_model[6])
    descs_bb_f1, descs_bb_miou = eval_bboxes(acc_gt[7], acc_model[7])


    with open(outfile, "a") as f:
        if pretty:
            f.write(f'Directories: {gt_dir} {model_dir}\n')
            f.write(f'Number of files: {count}\n')
            f.write(f'Valid files ratio: {valid / count}\n')
            f.write("##################\n")
            f.write("#### TEXT ####\n")
            f.write(f'Names CER: {names_cer}\n')
            f.write(f'Chapter numbers CER: {cnums_cer}\n')
            f.write(f'Page numbers CER: {pnums_cer}\n')
            f.write(f'Descriptions CER: {descs_cer}\n')
            f.write("\n#### BBOXES ####\n")
            f.write(f'Names bbox f1_score: {names_bb_f1}\n')
            f.write(f'Names bbox mean iou: {names_bb_miou}\n')
            f.write(f'Chapter numbers bbox f1_score: {cnums_bb_f1}\n')
            f.write(f'Chapter numbers bbox mean iou: {cnums_bb_miou}\n')
            f.write(f'Page numbers bbox f1_score: {pnums_bb_f1}\n')
            f.write(f'Page numbers bbox mean iou: {pnums_bb_miou}\n')
            f.write(f'Descriptions bbox f1_score: {descs_bb_f1}\n')
            f.write(f'Descriptions bbox mean iou: {descs_bb_miou}\n')
            f.write("\n#### JSON STRUCTURE ####\n")
            f.write(f'Structure chapters f1_score: {struct_f1_score}\n')
            f.write(f'Structure page number accuracy: {page_num_acc}\n')
            f.write(f'Structure chap number accuracy: {chap_num_acc}\n')
            f.write(f'Structure description accuracy: {desc_acc}\n')
            f.write("\n")
        else:
            f.write(f'{count}')
            f.write(f',{valid / count}')
            f.write(f',{names_cer}')
            f.write(f',{cnums_cer}')
            f.write(f',{pnums_cer}')
            f.write(f',{descs_cer}')
            f.write(f',{names_bb_f1}')
            f.write(f',{names_bb_miou}')
            f.write(f',{cnums_bb_f1}')
            f.write(f',{cnums_bb_miou}')
            f.write(f',{pnums_bb_f1}')
            f.write(f',{pnums_bb_miou}')
            f.write(f',{descs_bb_f1}')
            f.write(f',{descs_bb_miou}')
            f.write(f',{struct_f1_score}')
            f.write(f',{page_num_acc}')
            f.write(f',{chap_num_acc}')
            f.write(f',{desc_acc}\n')