from pydantic import BaseModel


class DeepSeekOCRModeConfig(BaseModel):
    """Configuration for a DeepSeek OCR inference mode.

    Attributes:
        base_size: The base resolution for image processing.
        image_size: The target image size for inference.
        crop_mode: Whether to use crop mode for large images.
    """

    base_size: int
    image_size: int
    crop_mode: bool


class DeepSeekOCRModeRegistry:
    """Registry of predefined DeepSeek OCR mode configurations.

    Attributes:
        TINY: Configuration for tiny mode (512x512, no crop).
        SMALL: Configuration for small mode (640x640, no crop).
        GUNDAM: Configuration for gundam mode (1024 base, 640 image, with crop).
        BASE: Configuration for base mode (1024x1024, no crop).
        LARGE: Configuration for large mode (1280x1280, no crop).
    """

    TINY = DeepSeekOCRModeConfig(base_size=512, image_size=512, crop_mode=False)
    SMALL = DeepSeekOCRModeConfig(base_size=640, image_size=640, crop_mode=False)
    GUNDAM = DeepSeekOCRModeConfig(base_size=1024, image_size=640, crop_mode=True)
    BASE = DeepSeekOCRModeConfig(base_size=1024, image_size=1024, crop_mode=False)
    LARGE = DeepSeekOCRModeConfig(base_size=1280, image_size=1280, crop_mode=False)
