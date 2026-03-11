import glob
import logging
import os.path
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional

from detector_parser import DetectorParser, AnnotatedBoundingBox

logger = logging.getLogger(__name__)

@dataclass
class ALTOWord:
    def __init__(self, x: float, y: float, width: float, height: float, content: str):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.content = content

    def __repr__(self):
        return f"ALTOWord(x={self.x}, y={self.y}, width={self.width}, height={self.height}, content={self.content})"

@dataclass
class ALTOPage:
    def __init__(self, page_id: str, xml_filename: str, width: float, height: float, words: List[ALTOWord]):
        self.page_id = page_id
        self.xml_filename = xml_filename
        self.width = width
        self.height = height
        self.words = words

    def __repr__(self):
        return f"ALTOPage(page_id={self.page_id}, xml_file_name={self.xml_filename}, width={self.width}, height={self.height})"

class ALTOExport:
    def __init__(self, alto_export_dir: str):
        self.alto_export_dir = alto_export_dir
        self.alto_pages = []

        self.parse_export()


    def parse_export(self):
        alto_xml_paths = list(glob.glob(self.alto_export_dir + "/*.xml"))
        for i, alto_xml_path in enumerate(alto_xml_paths):
            self.alto_pages.append(self.parse_page(alto_xml_path))
            if (i + 1) % 100 == 0:
                logger.info(f"Parsed {i + 1}/{len(alto_xml_paths)} pages")
        logger.info(f"Parsed {len(alto_xml_paths)}/{len(alto_xml_paths)} pages")



    def parse_page(self, alto_xml_path: str) -> ALTOPage:

        with open(alto_xml_path, "r") as f:
            xml_content = f.read()

        namespace = {'alto': 'http://www.loc.gov/standards/alto/ns-v2#'}
        tree = ET.ElementTree(ET.fromstring(xml_content))
        root = tree.getroot()

        # Extract Page ID, width, and height
        page_element = root.find(".//alto:Page", namespace)
        if page_element is None or not all(attr in page_element.attrib for attr in ["ID", "WIDTH", "HEIGHT"]):
            raise ValueError("Missing required attributes in <Page> element.")

        page_id = page_element.attrib["ID"]
        page_width = float(page_element.attrib["WIDTH"])
        page_height = float(page_element.attrib["HEIGHT"])

        alto_words = []

        for string_element in root.findall(".//alto:String", namespace):
            # Ensure all required attributes are present
            if not all(attr in string_element.attrib for attr in ["CONTENT", "HPOS", "VPOS", "WIDTH", "HEIGHT"]):
                logger.error(f"Missing required attributes in <String> element: {string_element.attrib}")
                raise ValueError("Missing required attributes in <String> element.")

            content = string_element.attrib["CONTENT"]
            x = float(string_element.attrib["HPOS"])
            y = float(string_element.attrib["VPOS"])
            width = float(string_element.attrib["WIDTH"])
            height = float(string_element.attrib["HEIGHT"])

            alto_word = ALTOWord(
                x=x,
                y=y,
                width=width,
                height=height,
                content=content
            )

            alto_words.append(alto_word)

        alto_page = ALTOPage(
            page_id=page_id,
            xml_filename=os.path.basename(alto_xml_path),
            width=page_width,
            height=page_height,
            words=alto_words
        )

        return alto_page


class ALTOMatchedDetection:
    def __init__(self,
                 detector_parser_annotated_bounding_box: AnnotatedBoundingBox,
                 alto_words: List[ALTOWord]):
        self.detector_parser_annotated_bounding_box = detector_parser_annotated_bounding_box
        self.alto_words = alto_words

    def get_class(self):
        return self.detector_parser_annotated_bounding_box.cls[0]

    def get_text(self):
        return " ".join([word.content for word in self.alto_words])

    def get_confidence(self):
        return self.detector_parser_annotated_bounding_box.conf

class ALTOMatchedPage:
    def __init__(self, detector_parser_page, alto_page, min_alto_word_area_in_detection_to_match=0.65):
        self.detector_parser_page = detector_parser_page
        self.alto_page = alto_page
        self.min_alto_word_area_in_detection_to_match = min_alto_word_area_in_detection_to_match
        self.matched_detections: Optional[List[ALTOMatchedDetection]] = None


    def match(self):
        matched_detections = []
        logger.info(
            f"detector parser page id: {self.detector_parser_page.id}, "
            f"detector parser page image filename: {self.detector_parser_page.image_filename}, "
            f"alto page xml filename: {self.alto_page.xml_filename}")
        for detector_parser_annotated_bounding_box in self.detector_parser_page.bounding_boxes:
            matched_words = []
            for alto_word in self.alto_page.words:
                logger.debug(f"matching {detector_parser_annotated_bounding_box.cls} -> {alto_word.content}")
                if self.is_detection_match(detector_parser_annotated_bounding_box, alto_word):
                    matched_words.append(alto_word)
                    logger.debug(f"Matched: {detector_parser_annotated_bounding_box.cls} -> {alto_word.content}")

            if len(matched_words) == 0:
                logger.warning(f"No words matched for detection -> "
                               f"detector parser page id: {self.detector_parser_page.id}, "
                               f"detector parser page image filename: {self.detector_parser_page.image_filename}, "
                               f"class: {detector_parser_annotated_bounding_box.cls}, "
                               f"confidence: {detector_parser_annotated_bounding_box.conf}, "
                               f"detection id: {detector_parser_annotated_bounding_box.id}, "
                               f"bbox: {detector_parser_annotated_bounding_box}")
                # continue
                
            matched_detection = ALTOMatchedDetection(detector_parser_annotated_bounding_box, matched_words)
            matched_detections.append(matched_detection)
        self.matched_detections = matched_detections

        for matched_detection in self.matched_detections:
            logger.info(
                f"{matched_detection.get_class()}: {matched_detection.get_text()} ({len(matched_detection.alto_words)} words)")
        logger.info("")



    def is_detection_match(self, detector_parser_annotated_bounding_box, alto_word):
        return self.is_bbox_inside(
            (detector_parser_annotated_bounding_box.x, detector_parser_annotated_bounding_box.y,
             detector_parser_annotated_bounding_box.width, detector_parser_annotated_bounding_box.height),
            (alto_word.x, alto_word.y, alto_word.width, alto_word.height),
            self.min_alto_word_area_in_detection_to_match
        )

    def is_bbox_inside(self, bbox1, bbox2, percentage=0.65):
        """
        Check if the second bounding box (bbox2) is inside the first bounding box (bbox1)
        by a specified percentage.

        Parameters:
        - bbox1: Tuple (x1, y1, width1, height1) for the first bounding box.
        - bbox2: Tuple (x2, y2, width2, height2) for the second bounding box.
        - percentage: Float value (0-1) specifying how much of bbox2 must be inside bbox1.

        Returns:
        - True if the specified percentage of bbox2 is inside bbox1, False otherwise.
        """
        x1, y1, width1, height1 = bbox1
        x2, y2, width2, height2 = bbox2

        logger.debug(f"bbox1: {bbox1}, bbox2: {bbox2}")

        # Define the corners of the boxes
        bbox1_xmin, bbox1_ymin = x1, y1
        bbox1_xmax, bbox1_ymax = x1 + width1, y1 + height1

        bbox2_xmin, bbox2_ymin = x2, y2
        bbox2_xmax, bbox2_ymax = x2 + width2, y2 + height2

        # Calculate the intersection box
        intersect_xmin = max(bbox1_xmin, bbox2_xmin)
        intersect_ymin = max(bbox1_ymin, bbox2_ymin)
        intersect_xmax = min(bbox1_xmax, bbox2_xmax)
        intersect_ymax = min(bbox1_ymax, bbox2_ymax)

        # Check if there is an intersection
        if intersect_xmin >= intersect_xmax or intersect_ymin >= intersect_ymax:
            return False

        # Compute the area of the intersection
        intersect_width = intersect_xmax - intersect_xmin
        intersect_height = intersect_ymax - intersect_ymin
        intersect_area = intersect_width * intersect_height

        # Compute the area of bbox2
        bbox1_area = width1 * height1
        bbox2_area = width2 * height2
        logger.debug(f"intersect_area: {intersect_area}, bbox2_area: {bbox2_area}")
        logger.debug(f"intersect_area / bbox2_area: {intersect_area / bbox2_area}")

        # Check if the required percentage of bbox2's area is inside bbox1
        return (intersect_area / bbox2_area) >= percentage or (intersect_area / bbox1_area) >= percentage


class ALTOMatch:
    def __init__(self, detector_parser: DetectorParser, alto_export_dir: str, min_alto_word_area_in_detection_to_match=0.65):
        self.detector_parser = detector_parser
        self.alto_export_dir = alto_export_dir
        self.min_alto_word_area_in_detection_to_match = min_alto_word_area_in_detection_to_match

        self.alto_export = ALTOExport(alto_export_dir)

        self.detector_parser_page_mapping = {}
        for page in self.detector_parser.annotated_pages:
            self.detector_parser_page_mapping[os.path.splitext(page.image_filename)[0]] = page

        self.alto_export_page_mapping = {}
        for page in self.alto_export.alto_pages:
            self.alto_export_page_mapping[os.path.splitext(page.xml_filename)[0]] = page

        logger.info(f"DetectorParser pages: {len(self.detector_parser_page_mapping)}")
        logger.info(f"ALTO pages: {len(self.alto_export_page_mapping)}")

        self.matched_pages: Optional[List[ALTOMatchedPage]] = None

    def match(self):
        matched_pages = []
        t_pages = 0
        m_pages = 0
        for image_filename, detector_parser_page in self.detector_parser_page_mapping.items():
            if image_filename in self.alto_export_page_mapping:
                matched_page = ALTOMatchedPage(detector_parser_page,
                                               self.alto_export_page_mapping[image_filename],
                                               self.min_alto_word_area_in_detection_to_match)
                matched_page.match()
                matched_pages.append(matched_page)
                m_pages += 1
            else:
                logger.warning(f"ALTO page not found for image filename: {image_filename}")
            t_pages += 1
        logger.info(f"Matched {m_pages}/{t_pages} pages")
        self.matched_pages = matched_pages



