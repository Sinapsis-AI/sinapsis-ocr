# -*- coding: utf-8 -*-
import gc
from typing import Literal

import cv2
import easyocr
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from sinapsis_core.data_containers.annotations import BoundingBox, ImageAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    TemplateAttributeType,
    UIPropertiesMetadata,
)
from sklearn.feature_extraction.text import strip_accents_unicode

from sinapsis_easyocr.helpers.tags import Tags


class ReaderParams(BaseModel):
    """Configuration parameters for initializing the EasyOCR Reader instance.

    This model holds settings related to model loading, hardware acceleration,
    and the core network architectures to be used by the OCR engine.
    """

    lang_list: list[str] = ["en"]
    gpu: bool = True
    model_storage_directory: str | None = None
    user_network_directory: str | None = None
    detect_network: Literal["craft", "dbnet18"] = "craft"
    recog_network: str = "standard"
    download_enabled: bool = True
    detector: bool = True
    recognizer: bool = True
    verbose: bool = True
    quantize: bool = True
    cudnn_benchmark: bool = False
    model_config = ConfigDict(extra="forbid")


class ReaderTextParams(BaseModel):
    """Configuration parameters for the `readtext` method.

    This model holds runtime settings that control how an image is processed,
    including detection thresholds and decoding options.
    """

    decoder: str = "greedy"
    beamWidth: int = 5
    batch_size: int = 1
    workers: int = 0
    detail: int = 1
    paragraph: bool = False
    min_size: int = 20
    contrast_ths: float = 0.1
    adjust_contrast: float = 0.5
    filter_ths: float = 0.003
    text_threshold: float = 0.7
    low_text: float = 0.4
    link_threshold: float = 0.4
    canvas_size: int = 2560
    mag_ratio: float = 1.0
    slope_ths: float = 0.1
    ycenter_ths: float = 0.5
    height_ths: float = 0.5
    width_ths: float = 0.5
    y_ths: float = 0.5
    x_ths: float = 1.0
    add_margin: float = 0.1
    threshold: float = 0.2
    bbox_min_score: float = 0.2
    bbox_min_size: int = 3
    max_candidates: int = 0
    model_config = ConfigDict(extra="forbid")


class EasyOCRAttributes(TemplateAttributes):
    """Attributes for EasyOCR template.

    reader_params (Dict[str, Any]): Params required to initialize easyocr.Reader class.
        Defaults to {"lang_list": ["en"]}. The full list of supported languages can be consulted in:
        https://www.jaided.ai/easyocr/
    read_text_params (dict[str, Any] | None): Params required by easyocr.Reader.readtext method. Defaults to None.
    get_full_text (bool): Enable the creation of a TextPacket containing all the individual text detections. Defaults to
        False.
    """

    reader_params: ReaderParams = Field(default_factory=ReaderParams)
    read_text_params: ReaderTextParams = Field(default_factory=ReaderTextParams)
    get_full_text: bool = False
    img_height: int = 1024
    img_width: int = 1024
    enable_img_resize: bool = False


class EasyOCR(Template):
    """
    This template uses EasyOCR to extract text from images, along with their
    bounding boxes and confidence scores. The extracted annotations are stored
    in `ImageAnnotations`, and optionally, full text can be stored in a `TextPacket`.

    Attributes:
        reader (easyocr.Reader): The OCR model initialized with the given settings.
    """

    AttributesBaseModel = EasyOCRAttributes
    UIProperties = UIPropertiesMetadata(
        category="OCR",
        output_type=OutputTypes.IMAGE,
        tags=[
            Tags.EASYOCR,
            Tags.DOCUMENT,
            Tags.IMAGE,
            Tags.OCR,
            Tags.TEXT,
            Tags.TEXT_RECOGNITION,
        ],
    )

    def __init__(self, attributes: TemplateAttributeType) -> None:
        super().__init__(attributes)
        self.reader = self.initialize_reader()

    def initialize_reader(self) -> easyocr.Reader:
        """Initializes and returns an instance of the easyocr.Reader

        Returns:
            easyocr.Reader: Instance of the easyocr.Reader with the selected
                `reader_params`.`
        """
        return easyocr.Reader(**self.attributes.reader_params.model_dump())

    def _get_bbox(self, bbox: np.ndarray, img: np.ndarray) -> BoundingBox:
        x1, y1 = np.min(bbox, axis=0)
        x2, y2 = np.max(bbox, axis=0)

        if self.attributes.enable_img_resize:
            original_shape = img.shape
            x_scale = original_shape[1] / self.attributes.img_width
            y_scale = original_shape[0] / self.attributes.img_height

            x1 *= x_scale
            x2 *= x_scale
            y1 *= y_scale
            y2 *= y_scale

        return BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

    def _parse_results(self, results: list[dict], img: np.ndarray) -> list[ImageAnnotations]:
        """
        Converts EasyOCR results into a list of `ImageAnnotations` objects.

        Args:
            results (list[dict]): The results from EasyOCR.

        Returns:
            list[ImageAnnotations]: The extracted annotations, including bounding boxes, text, and confidence scores.
        """
        annotations = []
        for result in results:
            text = result.get("text")
            conf_score = result.get("confident", 1.0)

            bbox = self._get_bbox(np.array(result.get("boxes")), img)

            ann = ImageAnnotations(
                label_str=strip_accents_unicode(text),
                bbox=bbox,
                confidence_score=conf_score,
                text=strip_accents_unicode(text),
            )
            annotations.append(ann)

        return annotations

    @staticmethod
    def _get_text_packet_from_annotations(annotations: list[ImageAnnotations]) -> TextPacket:
        """
        Generates a `TextPacket` from extracted OCR annotations.

        Args:
            annotations (list[ImageAnnotations]): The extracted OCR annotations.

        Returns:
            TextPacket: A text packet containing all the detected text from the image.
        """
        detected_text = [ann.text for ann in annotations]

        full_text = " ".join(detected_text)
        text_packet = TextPacket(
            content=full_text,
        )

        return text_packet

    def _resize_input_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.attributes.img_width, self.attributes.img_height))

    def _process_images(self, container: DataContainer) -> None:
        """
        Processes each image packet in the container, producing OCR results and storing them as annotations.

        Args:
            container (DataContainer): Data-container with image packets to be processed.
        """
        for image_packet in container.images:
            img = image_packet.content
            if self.attributes.enable_img_resize:
                img = self._resize_input_image(img)

            results = self.reader.readtext(img, **self.attributes.read_text_params.model_dump(), output_format="dict")
            annotations = self._parse_results(results, image_packet.content)

            if annotations and self.attributes.get_full_text:
                text_packet = self._get_text_packet_from_annotations(annotations)
                container.texts.append(text_packet)

            image_packet.annotations.extend(annotations)

    def execute(self, container: DataContainer) -> DataContainer:
        """
        Performs OCR detection to the image packets stored in the data-container. The extracted text and predicted
        annotations are added to the output data-container.

        Args:
            container (DataContainer): Input data-container with image packets.

        Returns:
            DataContainer: Output data-container with predicted OCR results.
        """

        if not container.images:
            log_msg = f"No images to process in {self.instance_name}, returning input DataContainer."
            self.logger.debug(log_msg)
        else:
            self._process_images(container)
        return container

    def reset_state(self, template_name: str | None = None) -> None:
        """Releases the EasyOCR model from memory and re-initializes it.

        Args:
            template_name (str | None, optional): The name of the template instance being reset. Defaults to None.
        """
        _ = template_name
        del self.reader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.reader = self.initialize_reader()
