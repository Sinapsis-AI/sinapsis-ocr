from enum import Enum


class Tags(Enum):
    """UI tags for categorizing DeepSeek OCR templates.

    Attributes:
        DEEPSEEK: Tag for DeepSeek-related templates.
        IMAGE: Tag for image processing templates.
        OCR: Tag for optical character recognition templates.
        TEXT: Tag for text-related templates.
        TEXT_RECOGNITION: Tag for text recognition templates.
    """

    DEEPSEEK = "deepseek"
    IMAGE = "image"
    OCR = "optical_character_recognition"
    TEXT = "text"
    TEXT_RECOGNITION = "text_recognition"
