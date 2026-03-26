from typing import Any

import torch
from sinapsis_core.data_containers.data_packet import (
    DataContainer,
    TextPacket,
)

from sinapsis_glm_ocr.templates.glm_ocr_base_inference import GLMOCRBaseInference


class GLMOCRBatchInference(GLMOCRBaseInference):
    """Template for GLM OCR Batch inference.

    This template uses the GLM OCR model to extract text from images.
    GLM-OCR is a multimodal OCR model built on the GLM-V encoder-decoder architecture
    that supports document parsing (text, formula, table) and structured information
    extraction via JSON schema prompts.
    """

    def decode(self, inputs: dict[str, Any], outputs: torch.Tensor) -> list[str]:
        """Decode the generated token ids into human-readable strings.

        Args:
            inputs (dict[str, Any]): The processed inputs containing the input token ids.
            outputs (torch.Tensor): The generated token ids to be decoded.

        Returns:
            list[str]: The decoded output texts for each batch item.
        """
        input_lengths = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_lengths:]
        output_texts = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)

        return output_texts

    def execute(self, container: DataContainer) -> DataContainer:
        """Execute OCR on all images in the container.

        Args:
            container: The data container with images to process.

        Returns:
            DataContainer: The container with text packets containing OCR results.
        """
        messages = [self.build_messages(image_packet=image_packet) for image_packet in container.images]
        inputs = self.apply_chat_template(messages=messages, padding=True)
        outputs = self.generate(inputs=inputs)
        texts = self.decode(inputs=inputs, outputs=outputs)
        text_packets = [TextPacket(content=text) for text in texts]
        container.texts.extend(text_packets)

        return container
