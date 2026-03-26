from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic.dataclasses import dataclass
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class DeepSeekOCRInitArgs(BaseModel):
    """Initialization arguments for the DeepSeek OCR model.

    Attributes:
        pretrained_model_name_or_path (Literal["deepseek-ai/DeepSeek-OCR", "deepseek-ai/DeepSeek-OCR-2"]): HuggingFace
        model identifier. Defaults to 'deepseek-ai/DeepSeek-OCR'.
        cache_dir (str): Directory to cache downloaded models.
        torch_dtype (Literal["float16", "bfloat16", "auto"] | torch.torch_dtype): Precision for model weights.
        attn_implementation (Literal["flash_attention_2", "eager"]: Attention implementation.
        trust_remote_code (Literal[True]): Whether to trust remote code from HuggingFace.
        use_safetensors (Literal[True]): Whether to use safetensors format.
    """

    pretrained_model_name_or_path: Literal["deepseek-ai/DeepSeek-OCR", "deepseek-ai/DeepSeek-OCR-2"] = (
        "deepseek-ai/DeepSeek-OCR"
    )
    cache_dir: str = SINAPSIS_CACHE_DIR
    torch_dtype: Literal["float16", "bfloat16", "auto"] | torch.dtype = "auto"
    attn_implementation: Literal["flash_attention_2", "eager"] = "flash_attention_2"
    trust_remote_code: Literal[True] = True
    use_safetensors: Literal[True] = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def resolve_dtype(self) -> "DeepSeekOCRInitArgs":
        """Resolve 'auto' torch_dtype to 'float16' or 'bfloat16' based on availability.

        Returns:
            DeepSeekOCRInitArgs: The validated instance with resolved torch_dtype.
        """
        if self.torch_dtype == "auto":
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif isinstance(self.torch_dtype, str):
            self.torch_dtype = _DTYPE_MAP.get(self.torch_dtype, self.torch_dtype)
        return self


@dataclass
class GroundingResult:
    """A single grounding result with label and bounding box coordinates.

    Attributes:
        label: The text label for this grounding region.
        coordinates: List of (x1, y1, x2, y2) tuples in normalized 0-999 range.
    """

    label: str
    coordinates: list[tuple[int, int, int, int]]
