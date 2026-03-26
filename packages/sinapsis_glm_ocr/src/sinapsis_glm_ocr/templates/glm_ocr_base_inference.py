import gc
from abc import abstractmethod
from typing import Any

import torch
from pydantic import Field
from sinapsis_core.data_containers.data_packet import (
    ImageColor,
    ImagePacket,
)
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    UIPropertiesMetadata,
)
from sinapsis_generic_data_tools.helpers.image_color_space_converter_cv import (
    convert_color_space_cv,
)
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

from sinapsis_glm_ocr.helpers.schemas import GLMOCRGenerationConfig, GLMOCRInitArgs
from sinapsis_glm_ocr.helpers.tags import Tags

MESSAGES_TYPE = list[dict[str, Any]] | list[list[dict[str, Any]]]


class GLMOCRBaseInferenceAttributes(TemplateAttributes):
    """Base Attributes for GLM OCR inference.

    Attributes:
        prompt: The prompt to send to the model.
        init_args: Initialization arguments for the model.
        generation_config: Generation configuration for inference.
    """

    prompt: str = "Text Recognition:"
    init_args: GLMOCRInitArgs = Field(default_factory=GLMOCRInitArgs)
    generation_config: GLMOCRGenerationConfig = Field(default_factory=GLMOCRGenerationConfig)


class GLMOCRBaseInference(Template):
    """Base Template for GLM OCR inference.

    This template uses the GLM OCR model to extract text from images.
    GLM-OCR is a multimodal OCR model built on the GLM-V encoder-decoder architecture
    that supports document parsing (text, formula, table) and structured information
    extraction via JSON schema prompts.
    """

    AttributesBaseModel = GLMOCRBaseInferenceAttributes
    UIProperties = UIPropertiesMetadata(
        category="OCR",
        output_type=OutputTypes.TEXT,
        tags=[
            Tags.GLM,
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
        """Load the model and processor from pretrained weights."""
        self.model = self._initialize_model()
        self.processor = AutoProcessor.from_pretrained(
            self.attributes.init_args.pretrained_model_name_or_path,
            cache_dir=self.attributes.init_args.cache_dir,
        )

    def _initialize_model(self) -> GlmOcrForConditionalGeneration:
        """Initialize and return the model with appropriate configuration.

        Note: Uses CUDA if available as GLM-OCR performs best on GPU.

        Returns:
            AutoModelForImageTextToText: The initialized model ready for inference.
        """
        model = GlmOcrForConditionalGeneration.from_pretrained(**self.attributes.init_args.model_dump())
        return model.eval()

    def build_messages(self, image_packet: ImagePacket) -> list[dict[str, Any]]:
        """Build the message structure for the model.

        GLM-OCR uses a chat-like message format with image and text content.

        Args:
            image_packet: The image packet containing the image to process.

        Returns:
            List of message dictionaries with image and text content.
        """
        converted_packet = convert_color_space_cv(image_packet=image_packet, desired_color_space=ImageColor.RGB)

        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": converted_packet.content,
                    },
                    {
                        "type": "text",
                        "text": self.attributes.prompt,
                    },
                ],
            }
        ]

    def apply_chat_template(self, messages: MESSAGES_TYPE, **kwargs: Any) -> dict[str, Any]:
        """Applies a chat template to the given messages and processes the input.

        Args:
            messages (MESSAGES_TYPE): The messages to which the chat template will be applied.
            **kwargs: Additional keyword arguments for processing.

        Returns:
            dict[str, Any]: The processed input as a dict.
        """
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", **kwargs
        ).to(self.model.device)

        return inputs

    def generate(self, inputs: dict[str, Any]) -> torch.Tensor:
        """Generate tokens from the given inputs.

        Args:
            inputs (dict[str, Any]): The processed inputs containing tokenized data.

        Returns:
            torch.Tensor: The generated token ids from the model.
        """
        return self.model.generate(
            **inputs,
            **self.attributes.generation_config.model_dump(),
        )

    @abstractmethod
    def decode(self, inputs: dict[str, Any], outputs: torch.Tensor) -> str | list[str]:
        """Decode the given inputs and outputs into a string or a list of strings.

        Subclasses must implement this method.

        Args:
            inputs (dict[str, Any]): A dictionary containing input data for decoding.
            outputs (torch.Tensor): A tensor containing the output data to be decoded.

        Returns:
            str | list[str]: The decoded result as a string or a list of strings.
        """

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
