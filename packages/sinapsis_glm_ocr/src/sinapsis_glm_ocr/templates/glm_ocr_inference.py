from typing import Any

import torch
from sinapsis_core.data_containers.data_packet import (
    DataContainer,
    TextPacket,
)

from sinapsis_glm_ocr.templates.glm_ocr_base_inference import GLMOCRBaseInference


class GLMOCRInference(GLMOCRBaseInference):
    """Template for GLM OCR inference.

    This template uses the GLM OCR model to extract text from images.
    GLM-OCR is a multimodal OCR model built on the GLM-V encoder-decoder architecture
    that supports document parsing (text, formula, table) and structured information
    extraction via JSON schema prompts.
    """

    def decode(self, inputs: dict[str, Any], outputs: torch.Tensor) -> str:
        """Decode the generated token ids into a human-readable string.

        Args:
            inputs (dict[str, Any]): The processed inputs containing the input token ids.
            outputs (torch.Tensor): The generated token ids to be decoded.

        Returns:
            str: The decoded output text.
        """
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)

        return output_text

    def execute(self, container: DataContainer) -> DataContainer:
        """Execute OCR on all images in the container.

        Args:
            container: The data container with images to process.

        Returns:
            DataContainer: The container with text packets containing OCR results.
        """
        for image_packet in container.images:
            messages = self.build_messages(image_packet=image_packet)
            inputs = self.apply_chat_template(messages=messages)
            outputs = self.generate(inputs=inputs)
            text = self.decode(inputs=inputs, outputs=outputs)
            container.texts.append(TextPacket(content=text))

        return container
