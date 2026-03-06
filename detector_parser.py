"""Definition of the export data file from Label-Studio. Running this file is for debug only.

Date -- 16.1.2025
Author -- Martin Kostelnik
"""


import argparse
import glob
import json
import logging
import cv2
import typing
from dataclasses import dataclass, field
from enum import Enum
import os
from typing import Optional
import shutil
from urllib.parse import unquote
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)


RawExport = list[dict[str, any]]

def load_export(
    export_path: str
) -> RawExport:
    with open(export_path, "r") as file:
        return json.load(file)


class AnnotatedObjectType(Enum):
    BOUNDING_BOX = "rectanglelabels"
    KEYPOINT = "keypointlabels"
    RELATION = "relation"


@dataclass
class AnnotatedBoundingBox:
    id: str
    cls: list[str]
    x: float
    y: float
    width: float
    height: float
    rotation: float
    conf: Optional[float] = None

    def __repr__(self):
        return f"AnnotatedBoundingBox(id={self.id}, cls={self.cls}, x={self.x}, y={self.y}, width={self.width}, height={self.height}, rotation={self.rotation}, conf={self.conf})"


@dataclass
class AnnotatedKeypoint:
    id: str
    cls: list[str]
    x: float
    y: float


@dataclass
class AnnotatedRelation:
    cls: list[str]
    from_id: str
    to_id: str


@dataclass
class AnnotatedPage:
    id: str
    width: Optional[float] = None
    height: Optional[float] = None
    image_filename: Optional[str] = None
    bounding_boxes: list[AnnotatedBoundingBox] = field(default_factory=list)
    relations: list[AnnotatedRelation] = field(default_factory=list)
    keypoints: list[AnnotatedKeypoint] = field(default_factory=list)

    def __len__(
        self,
    ) -> int:
        return len(self.bounding_boxes) + len(self.relations) + len(self.keypoints)
    
    def bbox_iterator(
        self,
    ):
        return iter(self.bounding_boxes)
    
    def relation_iterator(
        self,
    ):
        return iter(self.relations)
    
    def keypoint_iterator(
        self,
    ):
        return iter(self.keypoints)


class DetectorParser:
    DATA_PREFIX = ""
    def __init__(self):
        self.annotated_pages: Optional[list[AnnotatedPage]] = None
        self.class_mapping: dict[str, int] = {}

    # LABEL STUDIO PARSING
    ####################################################################################################
    def parse_label_studio(
        self,
        label_studio_export: RawExport | str,
        run_checks: bool = True,
        class_remapping: Optional[dict[str, str]] = None
    ):

        self.class_remapping = class_remapping

        if isinstance(label_studio_export, str):
            label_studio_export = load_export(label_studio_export)

        result = []

        for i, annotated_page in enumerate(label_studio_export):
            parsed_page = self.parse_label_studio_page(annotated_page)
            if parsed_page is None:
                continue
            result.append(parsed_page)

            if (i + 1) % 100 == 0:
                logging.info(f"Parsed {i + 1}/{len(result)} pages")

        logging.info(f"Parsed {len(result)}/{len(result)} pages")

        self.annotated_pages = result

        if run_checks:
            self.check_label_studio_export()
    
    def parse_label_studio_page(
        self,
        annotated_page: dict[str, any],
    ) -> Optional[AnnotatedPage]:

        try:
            page_width = float(annotated_page["annotations"][0]["result"][0]["original_width"])
            page_height = float(annotated_page["annotations"][0]["result"][0]["original_height"])
        except (IndexError, KeyError):
            logger.warning(f"Page {annotated_page['id']} does not contain any annotations.")
            return None

        parsed_page = AnnotatedPage(
            id=annotated_page["id"],
            width=page_width,
            height=page_height
        )

        logging.debug(f"Parsing task with ID: {parsed_page.id}")

        if len(annotated_page["annotations"]) > 1:
            logger.warning(f"Page {id} contains more than one annotation. Taking the first one.")

        annotation = annotated_page["annotations"][0]

        try_keys = ["image", "img"]
        for key in try_keys:
            if key in annotated_page["data"]:
                parsed_page.image_filename = unquote(os.path.basename(annotated_page["data"][key]))
                break
        else:
            for value in annotated_page["data"].values():
                if value.endswith(".jpg"):
                    parsed_page.image_filename = unquote(os.path.basename(value))
                    break

        for annotated_object in annotation["result"]:
            try:
                annotated_object_type = AnnotatedObjectType(annotated_object["type"])
            except ValueError:
                logger.warning(f"Unknown annotated object type: {annotated_object['type']} in {annotated_page['id']}.")
                continue

            match annotated_object_type:
                case AnnotatedObjectType.BOUNDING_BOX:
                    parsed_page.bounding_boxes.append(self.parse_label_studio_bounding_box(annotated_object))
                
                case AnnotatedObjectType.KEYPOINT:
                    parsed_page.keypoints.append(self.parse_label_studio_keypoint(annotated_object))
                
                case AnnotatedObjectType.RELATION:
                    parsed_page.relations.append(self.parse_label_studio_relation(annotated_object))

        return parsed_page

    def parse_label_studio_bounding_box(
        self,
        bbox: dict[str, any],
    ) -> AnnotatedBoundingBox:

        cls = bbox["value"]["rectanglelabels"]
        if self.class_remapping is not None:
            if cls and cls[0] in self.class_remapping:
                cls = [self.class_remapping[cls[0]]]

        bbox = AnnotatedBoundingBox(
            id=bbox["id"],
            cls=cls,
            x=(bbox["value"]["x"]  / 100.0) * bbox["original_width"],
            y=(bbox["value"]["y"]  / 100.0) * bbox["original_height"],
            width=(bbox["value"]["width"] / 100.0) * bbox["original_width"],
            height=(bbox["value"]["height"] / 100.0) * bbox["original_height"],
            rotation=bbox["value"]["rotation"],
        )

        for cls in bbox.cls:
            if cls not in self.class_mapping:
                self.class_mapping[cls] = max(self.class_mapping.values(), default=-1) + 1

        return bbox

    def parse_label_studio_keypoint(
        self,
        keypoint: dict[str, any],
    ) -> AnnotatedKeypoint:
        cls = keypoint["value"]["keypointlabels"]
        if self.class_remapping is not None:
            if cls[0] in self.class_remapping:
                cls = [self.class_remapping[cls[0]]]

        return AnnotatedKeypoint(
            id=keypoint["id"],
            cls=cls,
            x=(keypoint["value"]["x"] / 100.0) * keypoint["original_width"],
            y=(keypoint["value"]["y"] / 100.0) * keypoint["original_height"]
        )
    
    def parse_label_studio_relation(
        self,
        relation: dict[str, any],
    ) -> AnnotatedRelation:
        cls = []
        if "labels" in relation:
            if self.class_remapping is not None:
                for label in relation["labels"]:
                    if label in self.class_remapping:
                        cls.append(self.class_remapping[label])
                    else:
                        cls.append(label)
            else:
                cls = relation["labels"]

        return AnnotatedRelation(
            cls=cls,
            from_id=relation["from_id"],
            to_id=relation["to_id"],
        )

    def __len__(
        self,
    ) -> int:
        return len(self.annotated_pages)
    
    def __iter__(
        self,
    ):
        return iter(self.annotated_pages)

    def check_label_studio_export(
        self,
    ) -> None:
        for annotated_page in self.annotated_pages:
            for annotated_bbox in annotated_page.bbox_iterator():
                if annotated_bbox.width < 10:
                    logger.warning(
                        f"Bounding box of page {annotated_page.id} with id {annotated_bbox.id} and class {annotated_bbox.cls} has width < 10px ({annotated_bbox.width:.2f}px).")
                if annotated_bbox.height < 10:
                    logger.warning(
                        f"Bounding box of page {annotated_page.id} with id {annotated_bbox.id} and class {annotated_bbox.cls} has height < 10px ({annotated_bbox.height:.2f}px).")
                if annotated_bbox.width * annotated_bbox.height < 15 * 15:
                    logger.warning(
                        f"Bounding box of page {annotated_page.id} with id {annotated_bbox.id} and class {annotated_bbox.cls} has area < 15x15px ({annotated_bbox.width:.2f}x{annotated_bbox.height:.2f}px).")
                if annotated_bbox.width <= 0 or annotated_bbox.width > annotated_page.width:
                    logger.warning(f"Bounding box of page {annotated_page.id} with id {annotated_bbox.id} and class {annotated_bbox.cls} has invalid width: {annotated_bbox.width:.2f}px.")

                if annotated_bbox.height <= 0 or annotated_bbox.height > annotated_page.height:
                    logger.warning(f"Bounding box of page {annotated_page.id} with id {annotated_bbox.id} and class {annotated_bbox.cls} has invalid height: {annotated_bbox.height:.2f}px.")

                if annotated_bbox.x < 0 or annotated_bbox.x > annotated_page.width:
                    logger.warning(f"Bounding box of page {annotated_page.id} with id {annotated_bbox.id} and class {annotated_bbox.cls} has invalid x: {annotated_bbox.x:.2f}px.")

                if annotated_bbox.y < 0 or annotated_bbox.y > annotated_page.height:
                    logger.warning(f"Bounding box of page {annotated_page.id} with id {annotated_bbox.id} and class {annotated_bbox.cls} has invalid y: {annotated_bbox.y:.2f}px.")

                if annotated_bbox.cls is None or len(annotated_bbox.cls) == 0:
                    logger.warning(f"Bounding box of page {annotated_page.id} with id {annotated_bbox.id} and class {annotated_bbox.cls} has no class.")

            for annotated_keypoint in annotated_page.keypoint_iterator():
                if annotated_keypoint.x < 0 or annotated_keypoint.x > annotated_page.width:
                    logger.warning(f"Keypoint of page {annotated_page.id} with id {annotated_keypoint.id} and class {annotated_keypoint.cls} has invalid x: {annotated_keypoint.x:.2f}px.")

                if annotated_keypoint.y < 0 or annotated_keypoint.y > annotated_page.height:
                    logger.warning(f"Keypoint of page {annotated_page.id} with id {annotated_keypoint.id} and class {annotated_keypoint.cls} has invalid y: {annotated_keypoint.y:.2f}px.")

                if annotated_keypoint.cls is None or len(annotated_keypoint.cls) == 0:
                    logger.warning(f"Keypoint of page {annotated_page.id} with id {annotated_keypoint.id} and class {annotated_keypoint.cls} has no class.")

            for annotated_relation in annotated_page.relation_iterator():
                from_id, to_id = annotated_relation.from_id, annotated_relation.to_id

                object_ids = [bbox.id for bbox in annotated_page.bbox_iterator()] + [keypoint.id for keypoint in annotated_page.keypoint_iterator()]
                
                if from_id not in object_ids:
                    logger.warning(f"'from_id': {from_id} of relation in page {annotated_page.id} not found in objects.")

                if to_id not in object_ids:
                    logger.warning(f"'to_id': {to_id} of relation in page {annotated_page.id} not found in objects.")
    ####################################################################################################

    def prune_classes(
        self,
        classes_to_remove: list[str],
    ) -> None:
        for annotated_page in self.annotated_pages:
            bbox_ids_to_remove = [bbox.id for bbox in annotated_page.bounding_boxes if bbox.cls[0] in classes_to_remove]
            annotated_page.relations = [
                relation for relation in annotated_page.relations if 
                    relation.from_id not in bbox_ids_to_remove and 
                    relation.to_id not in bbox_ids_to_remove
            ]

            annotated_page.bounding_boxes = [bbox for bbox in annotated_page.bounding_boxes if bbox.cls[0] not in classes_to_remove]

    #YOLO PARSING
    ####################################################################################################
    def parse_yolo(self, yolo_dir: str, yolo_yaml_path: typing.Optional[str] = None,
                   image_dir: typing.Optional[str] = None,
                   default_confidence: typing.Optional[float] = None) -> None:
        """
        Parse YOLO format into the internal representation.

        Args:
            yolo_dir: Path to the directory with YOLO files -> cls x y w h conf [class_name], absolute coordinates, if image_dir specified relative coordinates.
            yolo_yaml_path: Path to the YAML file with class mapping, if specified cls is supposed to be id that will be
             converted to class name, otherwise the input is supposed to be in the cls x y w h conf class_name format.
            image_dir: Path to the directory with images.

        """
        if yolo_yaml_path is not None:
            with open(yolo_yaml_path, "r") as file:
                yolo_yaml = yaml.safe_load(file)
                self.class_mapping = {cls: id for id, cls in yolo_yaml["names"].items()}
                id_to_class = {id: cls for cls, id in self.class_mapping.items()}


        self.annotated_pages = []
        yolo_paths = list(glob.glob(os.path.join(yolo_dir, "*.txt")))
        for i, yolo_page in enumerate(yolo_paths):
            with open(yolo_page, "r") as file:
                lines = file.readlines()

            page_id = str(os.path.basename(yolo_page).replace(".txt", ""))
            annotated_page = AnnotatedPage(id=page_id,
                                           image_filename=page_id + ".jpg")

            for line in lines:
                try:
                    cls_id, x, y, width, height, conf = [float(value) for value in line.split(" ")[:6]]
                except ValueError:
                    cls_id, x, y, width, height = [float(value) for value in line.split(" ")[:5]]
                    conf = default_confidence
                if image_dir is not None:
                    # Convert to absolute coordinates if image_dir is specified
                    image_path = os.path.join(image_dir, annotated_page.image_filename)
                    if not os.path.exists(image_path):
                        logger.warning(f"Image {annotated_page.image_filename} not found in {image_dir}. Skipping.")
                        continue
                    image_size = cv2.imread(image_path).shape[:2]  # (height, width, channels)
                    annotated_page.height = image_size[0]
                    annotated_page.width = image_size[1]
                    x *= annotated_page.width
                    y *= annotated_page.height
                    width *= annotated_page.width
                    height *= annotated_page.height
                    x -= width // 2
                    y -= height // 2
                else:
                    x -= width // 2
                    y -= height // 2
                cls_id = int(cls_id)
                if yolo_yaml_path is not None:
                    cls = id_to_class[cls_id]
                else:
                    cls = " ".join(line.split(" ")[6:]).strip()
                    self.class_mapping[cls] = cls_id
                annotated_bbox = AnnotatedBoundingBox(id=str(len(annotated_page.bounding_boxes)),
                                                      cls=[cls],
                                                      x=x,
                                                      y=y,
                                                      width=width,
                                                      height=height,
                                                      rotation=0,
                                                      conf=conf)
                annotated_page.bounding_boxes.append(annotated_bbox)

            self.annotated_pages.append(annotated_page)
            if (i + 1) % 100 == 0:
                logger.info(f"Parsed {i + 1}/{len(yolo_paths)} pages")
        logger.info(f"Parsed {len(yolo_paths)}/{len(yolo_paths)} pages")
    ####################################################################################################


    #YOLO EXPORT
    ####################################################################################################
    def export_yolo(
        self,
        output_folder: str,
        image_folder: Optional[str] = None,
    ) -> None:
        yolo_path = os.path.join(output_folder, "labels")
        os.makedirs(yolo_path, exist_ok=True)

        if image_folder is not None:
            tgt_image_folder = os.path.join(output_folder, "images")
            os.makedirs(tgt_image_folder, exist_ok=True)

        for annotated_page in self.annotated_pages:
            result_str = "\n".join([self.get_bbox_yolo_string(bbox, annotated_page) for bbox in annotated_page.bbox_iterator()]) + "\n"
            output_filename = annotated_page.image_filename[:-len(".jpg")] + ".txt" if annotated_page.image_filename is not None else annotated_page.id
            file_output_path = os.path.join(yolo_path, output_filename)

            with open(file_output_path, "w") as file:
                print(result_str, file=file, end="")

            if image_folder is not None:
                # find image in a non-flat directory structure
                for root, _, files in os.walk(image_folder):
                    if annotated_page.image_filename in files:
                        src_image_path = os.path.join(root, annotated_page.image_filename)
                        shutil.copy(src_image_path, tgt_image_folder)
                        break

                    # check if the image already is in the target folder
                    if annotated_page.image_filename in os.listdir(tgt_image_folder):
                        logger.warning("Image with the same name found in the target folder {image_p}.")
                        break
                else:
                    logger.warning(f"Image {annotated_page.image_filename} not found in {image_folder}.")
                    continue


        # Create mapping YAML
        mapping_str = f"path: .\ntrain: .\nval: .\n\nnc: {len(self.class_mapping)}\n\nnames:\n    " + "\n    ".join([f"{id}: {cls}" for cls, id in self.class_mapping.items()]) + "\n"
        yolo_path = os.path.join(output_folder, "data.yaml")
        with open(yolo_path, "w") as file:
            print(mapping_str, file=file, end="")

    def get_bbox_yolo_string(
        self,
        bbox: AnnotatedBoundingBox,
        page: AnnotatedPage
    ) -> str:
        if len(bbox.cls) > 1:
            logger.warning(f"Bounding box with id {self.id} has more than one class. Taking the first one.")

        yolo_bbox_x = (bbox.x + bbox.width / 2) / page.width
        yolo_bbox_y = (bbox.y + bbox.height / 2) / page.height
        yolo_bbox_width = bbox.width / page.width
        yolo_bbox_height = bbox.height / page.height

        return f"{self.class_mapping[bbox.cls[0]]} {yolo_bbox_x} {yolo_bbox_y} {yolo_bbox_width} {yolo_bbox_height}"
    ####################################################################################################

    # COCO EXPORT
    ####################################################################################################
    def export_coco(
        self,
        output_path: str,
        description: str = "Detection dataset.",
        version: str = "1.0",
        include_relations: bool = False,
        include_keypoints: bool = False,
        ignore_classes: list[str] = [],
        indent: int = 2,
    ) -> None:
        if include_keypoints:
            raise NotImplementedError("Keypoints are not yet supported.")

        coco = {
            "info": {
                "description": description,
                "version": version,
                "year": datetime.now().year,
            },
            "annotations": [],
            "images": [],
            "categories": [],
        }

        annotation_idx = 0
        relation_idx = 0
        id_mapping = {}

        for page_idx, annotated_page in enumerate(self.annotated_pages):
            image_entry = {
                "id": page_idx,
                "file_name": annotated_page.image_filename,
                "width": int(annotated_page.width),
                "height": int(annotated_page.height),
            }
            coco["images"].append(image_entry)

            bbox_mapping = {}
            for bbox in annotated_page.bbox_iterator():
                bbox_mapping[bbox.id] = bbox
                if bbox.cls[0] in ignore_classes:
                    continue

                annotation_entry = {
                    "id": annotation_idx,
                    "image_id": page_idx,
                    "category_id": self.class_mapping[bbox.cls[0]],
                    "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                    "area": bbox.width * bbox.height,
                    "iscrowd": 0,
                    "segmentation": [],
                }
                coco["annotations"].append(annotation_entry)
                id_mapping[bbox.id] = annotation_idx
                annotation_idx += 1

            if include_relations:
                coco["relations"] = [] if "relations" not in coco else coco["relations"]

                for relation in annotated_page.relation_iterator():
                    from_bbox = bbox_mapping[relation.from_id]
                    to_bbox = bbox_mapping[relation.to_id]
                    
                    if from_bbox.cls[0] in ignore_classes or to_bbox.cls[0] in ignore_classes:
                        continue

                    relation_entry = {
                        "id": relation_idx,
                        "image_id": page_idx,
                        "from_annotation_id": id_mapping[relation.from_id],
                        "to_annotation_id": id_mapping[relation.to_id],
                    }
                    coco["relations"].append(relation_entry)
                    relation_idx += 1

        for cls, id in self.class_mapping.items():
            if cls in ignore_classes:
                continue

            category_entry = {
                "id": id,
                "name": cls,
                "supercategory": "object",
            }
            coco["categories"].append(category_entry)

        with open(output_path, "w") as file:
            json.dump(coco, file, indent=indent)
    ####################################################################################################


def parse_arguments():
    parser = argparse.ArgumentParser(description="Classes and functions for parsing Label-Studio exports.")

    parser.add_argument("-e", "--export", type=str, required=True, help="Path to the Label-Studio export.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(level=logging.DEBUG)

    export_path = args.export
    export = DetectorParser()
    export.parse_label_studio(export_path, run_checks=True)

    print(f"Export contains {len(export)} annotated pages.")

    first_page = export.annotated_pages[0]
    print(f"First page contains {len(first_page)} annotated objects.")

    if len(first_page.bounding_boxes) > 0:
        first_bbox = first_page.bounding_boxes[0]
        print(f"First bounding box of first page: {first_bbox}")

    if len(first_page.keypoints) > 0:
        first_keypoint = first_page.keypoints[0]
        print(f"First keypoint of first page: {first_keypoint}")

    if len(first_page.relations) > 0:
        first_relation = first_page.relations[0]
        print(f"First relation of first page: {first_relation}")
    