from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR

_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16}


class GLMOCRInitArgs(BaseModel):
    """Initialization arguments for the GLM OCR model.

    Attributes:
        pretrained_model_name_or_path (str): HuggingFace model identifier or local path.
            Defaults to "zai-org/GLM-OCR".
        cache_dir (str): Directory to cache downloaded models. Defaults to SINAPSIS_CACHE_DIR.
        torch_dtype (Literal["float16", "bfloat16", "auto"] | torch.dtype): Precision for
            model weights. Defaults to "auto" (resolves to bfloat16 if supported, else float16).
        attn_implementation (Literal["kernels-community/flash-attn2", "kernels-community/paged-attention"]):
            Attention implementation. Defaults to "kernels-community/flash-attn2".
        device_map (Literal["auto", "balanced", "balanced_low_0", "sequential"] | str):
            Device mapping for model loading. Defaults to "auto".
    """

    pretrained_model_name_or_path: str = "zai-org/GLM-OCR"
    cache_dir: str = SINAPSIS_CACHE_DIR
    torch_dtype: Literal["float16", "bfloat16", "auto"] | torch.dtype = "auto"
    attn_implementation: Literal["kernels-community/flash-attn2", "kernels-community/paged-attention"] = (
        "kernels-community/flash-attn2"
    )
    device_map: Literal["auto", "balanced", "balanced_low_0", "sequential"] | str = "auto"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def resolve_torch_dtype(self) -> "GLMOCRInitArgs":
        """Resolve 'auto' torch_dtype to 'float16' or 'bfloat16' based on availability.

        Returns:
            GLMOCRInitArgs: The validated instance with resolved torch_dtype.
        """
        if self.torch_dtype == "auto":
            self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif isinstance(self.torch_dtype, str):
            self.torch_dtype = _DTYPE_MAP.get(self.torch_dtype, self.torch_dtype)
        return self


class GLMOCRGenerationConfig(BaseModel):
    """Generation configuration for GLM OCR inference.

    Attributes:
        max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 8192.
        min_new_tokens (int): Minimum number of new tokens to generate. Defaults to 1.
        do_sample (bool): Whether to use sampling for generation. Defaults to False.
        repetition_penalty (float): Penalty for repeating tokens (1.0 = no penalty).
            Defaults to 1.0.
        length_penalty (float): Penalty for sequence length (1.0 = no penalty).
            Defaults to 1.0.
    """

    max_new_tokens: int = 8192
    min_new_tokens: int = 1
    do_sample: bool = False
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
