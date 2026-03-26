import importlib
from collections.abc import Callable

_root_lib_path = "sinapsis_glm_ocr.templates"

_template_lookup = {
    "GLMOCRInference": f"{_root_lib_path}.glm_ocr_inference",
    "GLMOCRBatchInference": f"{_root_lib_path}.glm_ocr_batch_inference",
}


def __getattr__(name: str) -> Callable:
    """Retrieves a callable template by its name.

    Args:
        name (str): The name of the template to retrieve.

    Returns:
        Callable: The callable template associated with the given name.

    Raises:
        AttributeError: If the template name is not found in the lookup.
    """
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
