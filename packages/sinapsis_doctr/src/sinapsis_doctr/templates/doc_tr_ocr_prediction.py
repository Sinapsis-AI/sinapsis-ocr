# -*- coding: utf-8 -*-
from typing import Any, Literal, TypeAlias

import numpy as np
from doctr.io.elements import Document
from doctr.models import ocr_predictor
from pydantic.dataclasses import dataclass
from sinapsis_core.data_containers.annotations import BoundingBox, ImageAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sinapsis_data_visualization.helpers.detection_utils import bbox_xyxy_to_xywh

DetectionArchitectures: TypeAlias = Literal[
    "db_resnet34",
    "db_resnet50",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50fast_tiny",
    "fast_small",
    "fast_base",
]
RecognitionArchitectures: TypeAlias = Literal[
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "parseq",
]

FloatGeometry: TypeAlias = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class ElementProperty:
    """
    These constants represent the keys used in the dictionary representation
    of document elements returned by docTR.

    Attributes:
        GEOMETRY (str): Key for accessing the bounding box coordinates.
        CONFIDENCE (str): Key for accessing the confidence score.
        VALUE (str): Key for accessing the recognized text value of a word.
        TYPE (str): Key for accessing the type of the artefact.
    """

    GEOMETRY = "geometry"
    CONFIDENCE = "confidence"
    VALUE = "value"
    TYPE = "type"


@dataclass(frozen=True)
class DocumentElement:
    """
    Hierarchical elements of the Document structure in docTR's output.

    Attributes:
        PAGES (str): Key for accessing the pages list.
        BLOCKS (str): Key for accessing the blocks list within a page.
        LINES (str): Key for accessing the lines list within a block.
        WORDS (str): Key for accessing the words list within a line.
        ARTEFACTS (str): Key for accessing the artefacts list within a block.
    """

    PAGES = "pages"
    BLOCKS = "blocks"
    LINES = "lines"
    WORDS = "words"
    ARTEFACTS = "artefacts"


class DocTROCRPrediction(Template):
    """
    Template for Optical Character Recognition (OCR) prediction using docTR.

    This class leverages docTR's OCR models to extract text, bounding boxes,
    and confidence scores from images. The extracted information is stored
    as annotations to each image packet in the input DataContainer.

    docTR uses a two-stage approach for OCR:
    1. Text detection: Localizing words in the document
    2. Text recognition: Identifying characters in each detected word

    The template allows configuration of both detection and recognition architectures,
    as well as various processing options.
    """

    class AttributesBaseModel(TemplateAttributes):
        """
        Configuration attributes for the OCR model.

        Attributes:
            recognized_characters_as_labels (bool): Whether to use recognized text as label strings. Defaults to False.
            artefact_type_as_labels (bool): Whether to use artefact types as label strings. Defaults to False.
            det_arch (DETECTION_ARCHITECTURES): Detection architecture to use. Defaults to "fast_base".
                Options include db_resnet50, linknet_resnet18, fast_base, etc.
            reco_arch (RECOGNITION_ARCHITECTURES): Recognition architecture to use. Defaults to "crnn_vgg16_bn".
                Options include crnn_vgg16_bn, master, parseq, etc.
            pretrained (bool): Whether to use pretrained models. Defaults to True.
            pretrained_backbone (bool): Whether to use pretrained backbones. Defaults to True.
            assume_straight_pages (bool): Whether to assume pages are straight (not rotated). Defaults to True.
            preserve_aspect_ratio (bool): Whether to preserve aspect ratio during preprocessing. Defaults to True.
            symmetric_pad (bool): Whether to use symmetric padding. Defaults to True.
            export_as_straight_boxes (bool): Whether to export rotated boxes as straight boxes. Defaults to False.
            detect_orientation (bool): Whether to detect page orientation. Defaults to False.
            straighten_pages (bool): Whether to straighten pages before processing. Defaults to False.
            detect_language (bool): Whether to detect language of the text. Defaults to False.
        """

        recognized_characters_as_labels: bool = False
        artefact_type_as_labels: bool = False
        det_arch: DetectionArchitectures = "fast_base"
        reco_arch: RecognitionArchitectures = "crnn_vgg16_bn"
        pretrained: bool = True
        pretrained_backbone: bool = True
        assume_straight_pages: bool = True
        preserve_aspect_ratio: bool = True
        symmetric_pad: bool = True
        export_as_straight_boxes: bool = False
        detect_orientation: bool = False
        straighten_pages: bool = False
        detect_language: bool = False

    UIProperties = UIPropertiesMetadata(category="OCR", output_type=OutputTypes.MULTIMODAL)

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.model = ocr_predictor(
            **self.attributes.model_dump(
                exclude={
                    "metadata",
                    "recognized_characters_as_labels",
                    "artefact_type_as_labels",
                }
            )
        )

    @staticmethod
    def denormalize_geometry(geometry: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
        """
        Converts normalized geometry coordinates into pixel-space coordinates.

        Args:
            geometry (np.ndarray): Normalized bounding box coordinates.
            image_height (int): Height of the image in pixels.
            image_width (int): Width of the image in pixels.

        Returns:
            np.ndarray: Denormalized bounding box (x_min, y_min, x_max, y_max).
        """

        denorm_geometry = geometry * np.array([image_width, image_height, image_width, image_height])

        return denorm_geometry

    @staticmethod
    def preprocess_geometry(
        geometry: FloatGeometry,
    ) -> np.ndarray:
        """
        Converts the nested tuple geometry representation into a numpy array.

        Args:
            geometry (FloatGeometry): Nested coordinate format.

        Returns:
            np.ndarray: Coordinates in numpy array format.
        """
        return np.array(geometry).reshape(-1)

    @staticmethod
    def make_bbox(
        geometry: np.ndarray,
    ) -> BoundingBox:
        """
        Converts the geometry in xyxy format to xywh format and then into
        a BoundingBox object.

        Args:
            geometry (np.ndarray): (x_min, y_min, x_max, y_max).

        Returns:
            BoundingBox: The bounding box object in xywh.
        """
        xywh = bbox_xyxy_to_xywh(geometry)
        bbox = BoundingBox(
            x=xywh[0],
            y=xywh[1],
            w=xywh[2],
            h=xywh[3],
        )
        return bbox

    def get_word_label_str(self, word: dict[str, Any]) -> str | None:
        """
        Extracts a label string from a word object if configured.

        Args:
            word (dict[str, Any]): The word object from OCR output.

        Returns:
            str | None: The recognized word or None if labels are disabled.
        """

        return word[ElementProperty.VALUE] if self.attributes.recognized_characters_as_labels else None

    def get_artefact_label_str(self, artefact: dict[str, Any]) -> str | None:
        """
        Extracts a label string from an artefact object if configured.

        Args:
            artefact (dict[str, Any]): The artefact object from OCR output.

        Returns:
            str | None: The artefact type or None if labels are disabled.
        """

        return artefact[ElementProperty.TYPE] if self.attributes.artefact_type_as_labels else None

    def update_anns_from_artefact(
        self, artefact: dict[str, Any], image_height: int, image_width: int, anns: list[ImageAnnotations]
    ) -> None:
        """
        Create an ImageAnnotation object from the artefact elements and appends it to the annotations list.


        Args:
            artefact (dict[str, Any]): Dictionary containing artefact elements like geometry, label and confidence.
            image_height (int): Original image height.
            image_width (int): Original image width.
            anns (list[ImageAnnotations]): List of image annotations.
        """
        artefact_geometry = artefact.get(ElementProperty.GEOMETRY)

        if artefact_geometry is not None:
            denorm_geometry = self.denormalize_geometry(
                self.preprocess_geometry(artefact_geometry),
                image_height,
                image_width,
            )
            ann = ImageAnnotations(
                label_str=self.get_artefact_label_str(artefact),
                bbox=self.make_bbox(denorm_geometry),
                confidence_score=artefact.get(ElementProperty.CONFIDENCE),
            )
            anns.append(ann)

    def update_anns_from_word(
        self, word: dict[str, Any], image_height: int, image_width: int, anns: list[ImageAnnotations]
    ) -> None:
        """
        Create an ImageAnnotation object from the word elements and appends it to the annotations list.

        Args:
            word (dict[str, Any]): Dictionary containing word elements like geometry, label, confidence and value.
            image_height (int): Original image height.
            image_width (int): Original image width.
            anns (list[ImageAnnotations]): List of image annotations.
        """
        word_geometry = word.get(ElementProperty.GEOMETRY)
        if word_geometry is not None:
            denorm_geometry = self.denormalize_geometry(
                self.preprocess_geometry(word_geometry),
                image_height,
                image_width,
            )
            ann = ImageAnnotations(
                label_str=self.get_word_label_str(word),
                bbox=self.make_bbox(denorm_geometry),
                confidence_score=word.get(ElementProperty.CONFIDENCE),
                text=word.get(ElementProperty.VALUE),
            )
            anns.append(ann)

    def parse_document(self, document: Document, image_height: int, image_width: int) -> list[ImageAnnotations]:
        """
        Parses a DocTR Document object by first converting it into a dictionary
        structure which is then iterated over to extract the OCR results into ImageAnnotations objects.

        Args:
            document (Document): The OCR result as a Doctr Document object.
            image_height (int): The height of the document to parse.
            image_width (int): The width of the document to parse.

        Returns:
            list[ImageAnnotations]: Extracted annotations from the document.
        """
        anns: list[ImageAnnotations] = []
        document_dict = document.export()

        if not document_dict:
            self.logger.debug("Returning empty annotations list due to empty document dict")
            return anns

        for page in document_dict.get(DocumentElement.PAGES, {}):
            for block in page.get(DocumentElement.BLOCKS, {}):
                for line in block.get(DocumentElement.LINES, {}):
                    for word in line.get(DocumentElement.WORDS, {}):
                        self.update_anns_from_word(word, image_height, image_width, anns)

                    for artefact in block.get(DocumentElement.ARTEFACTS, {}):
                        self.update_anns_from_artefact(artefact, image_height, image_width, anns)

        return anns

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Applies OCR to images in the given DataContainer and stores their results as annotations in their
          corresponding packets.

        Args:
            container (DataContainer): The input data container with images.

        Returns:
            DataContainer: The same container with OCR annotations added to the image packets.
        """
        for image_packet in container.images:
            document = self.model([image_packet.content])
            anns = self.parse_document(
                document,
                image_height=image_packet.shape[0],
                image_width=image_packet.shape[1],
            )
            image_packet.annotations.extend(anns)
        return container
