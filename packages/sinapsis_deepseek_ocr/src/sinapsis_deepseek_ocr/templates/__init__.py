import importlib
from collections.abc import Callable

_root_lib_path = "sinapsis_deepseek_ocr.templates"

_template_lookup = {
    "DeepSeekOCRInference": f"{_root_lib_path}.deepseek_ocr_inference",
}


def __getattr__(name: str) -> Callable:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
