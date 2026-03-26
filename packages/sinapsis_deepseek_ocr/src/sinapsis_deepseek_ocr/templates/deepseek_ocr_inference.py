import gc
import os
import tempfile
from typing import Any, Literal

import cv2
import torch
from pydantic import Field
from sinapsis_core.data_containers.annotations import BoundingBox, ImageAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer, ImageColor, ImagePacket, TextPacket
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    UIPropertiesMetadata,
)
from sinapsis_generic_data_tools.helpers.image_color_space_converter_cv import convert_color_space_cv
from transformers import AutoModel, AutoTokenizer

from sinapsis_deepseek_ocr.helpers.bbox_utils import denormalize_bbox
from sinapsis_deepseek_ocr.helpers.grounding_parser import parse_grounding_output
from sinapsis_deepseek_ocr.helpers.mode_registry import DeepSeekOCRModeRegistry
from sinapsis_deepseek_ocr.helpers.schemas import DeepSeekOCRInitArgs, GroundingResult
from sinapsis_deepseek_ocr.helpers.tags import Tags


class DeepSeekOCRInferenceAttributes(TemplateAttributes):
    """Attributes for DeepSeek OCR inference.

    Attributes:
        prompt: The prompt to send to the model (without <image> or <|grounding|> tags).
        enable_grounding: Whether to enable grounding for bounding box extraction.
        mode: The inference mode configuration to use.
        init_args: Initialization arguments for the model.
    """

    prompt: str = "OCR the image."
    enable_grounding: bool = False
    mode: Literal["tiny", "small", "gundam", "base", "large"] = "base"
    init_args: DeepSeekOCRInitArgs = Field(default_factory=DeepSeekOCRInitArgs)


class DeepSeekOCRInference(Template):
    """Template for DeepSeek OCR inference.

    This template uses the DeepSeek OCR model to extract text from images.

    Attributes:
        MODE_REGISTRY: Registry of available inference mode configurations.
        AttributesBaseModel: The Pydantic model for template attributes.
        UIProperties: UI metadata for the template.
    """

    MODE_REGISTRY = DeepSeekOCRModeRegistry
    AttributesBaseModel = DeepSeekOCRInferenceAttributes
    UIProperties = UIPropertiesMetadata(
        category="OCR",
        output_type=OutputTypes.MULTIMODAL,
        tags=[
            Tags.DEEPSEEK,
            Tags.IMAGE,
            Tags.OCR,
            Tags.TEXT,
            Tags.TEXT_RECOGNITION,
        ],
    )

    def __init__(self, attributes: TemplateAttributes) -> None:
        super().__init__(attributes)
        self.initialize()

    def initialize(self) -> None:
        """Load the model and tokenizer from pretrained weights."""
        self.model = self._initialize_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.attributes.init_args.pretrained_model_name_or_path,
            cache_dir=self.attributes.init_args.cache_dir,
            trust_remote_code=self.attributes.init_args.trust_remote_code,
        )
        self.mode_config = self._get_mode_config()
        self.full_prompt = self._build_prompt()

    def _initialize_model(self) -> AutoModel:
        """Initialize and return the model."""
        model = AutoModel.from_pretrained(**self.attributes.init_args.model_dump())
        return model.eval().cuda()

    def _get_mode_config(self) -> dict[str, Any]:
        """Get the mode configuration for the current inference mode.

        Returns:
            dict[str, Any]: The mode configuration as a dictionary.
        """
        mode_config = getattr(self.MODE_REGISTRY, self.attributes.mode.upper())
        return mode_config.model_dump()

    def _build_prompt(self) -> str:
        """Build the full prompt with appropriate tags.

        Returns:
            Full prompt with <image> and optional <|grounding|> tags.
        """
        if self.attributes.enable_grounding:
            return f"<image>\n<|grounding|>{self.attributes.prompt}"
        return f"<image>\n{self.attributes.prompt}"

    def infer(self, image_packet: ImagePacket) -> str:
        """Run OCR inference on an image packet.

        Args:
            image_packet: The image packet containing the image to process.

        Returns:
            str: The raw OCR result from the model.
        """
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)

        try:
            image_packet = convert_color_space_cv(image_packet=image_packet, desired_color_space=ImageColor.BGR)
            cv2.imwrite(tmp_path, image_packet.content)

            raw_result = self.model.infer(
                tokenizer=self.tokenizer,
                prompt=self.full_prompt,
                image_file=tmp_path,
                output_path=self.attributes.init_args.cache_dir,
                eval_mode=True,
                save_results=False,
                **self.mode_config,
            )

            return raw_result

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def grounding_to_annotations(results: list[GroundingResult], width: int, height: int) -> list[ImageAnnotations]:
        """Convert GroundingResult list to ImageAnnotations.

        Args:
            results: List of parsed grounding results.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            List of ImageAnnotations with denormalized bounding boxes.
        """
        annotations: list[ImageAnnotations] = []

        for result in results:
            for coords in result.coordinates:
                x1, y1, x2, y2 = denormalize_bbox(coords, width, height)
                bbox = BoundingBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
                ann = ImageAnnotations(
                    label_str=result.label,
                    bbox=bbox,
                    text=result.label,
                )
                annotations.append(ann)

        return annotations

    @staticmethod
    def clear_memory() -> None:
        """Clear memory to free up resources.

        Performs garbage collection and clears GPU memory if available.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def reset_state(self, template_name: str | None = None) -> None:
        """Reset the template state by reinitializing the model.

        Args:
            template_name: Optional template name (unused, for interface compatibility).
        """
        _ = template_name

        if hasattr(self, "model"):
            del self.model

        self.clear_memory()
        self.initialize()
        self.logger.info(f"Reset template instance `{self.instance_name}`")

    def execute(self, container: DataContainer) -> DataContainer:
        """Execute OCR on all images in the container.

        Args:
            container: The data container with images to process.

        Returns:
            DataContainer: The container with annotations or text packets.
        """
        for image_packet in container.images:
            raw_result = self.infer(image_packet)

            if self.attributes.enable_grounding:
                grounding_results = parse_grounding_output(raw_result)
                height, width = image_packet.shape[:2]
                annotations = self.grounding_to_annotations(grounding_results, width, height)
                image_packet.annotations.extend(annotations)
            else:
                container.texts.append(TextPacket(content=raw_result))

        return container
