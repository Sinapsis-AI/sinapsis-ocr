"""Parser for DeepSeek-OCR grounding output."""

import re

from sinapsis_deepseek_ocr.helpers.schemas import GroundingResult

GROUNDING_PATTERN = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"


def _parse_coordinates(coords_str: str) -> list[tuple[int, int, int, int]] | None:
    """Parse coordinates string, returning None on failure."""
    try:
        coords_list = eval(coords_str)
        if isinstance(coords_list, list):
            return [tuple(c) for c in coords_list]
        return None
    except (SyntaxError, ValueError):
        return None


def parse_grounding_output(text: str) -> list[GroundingResult]:
    """Parse grounding tags from model output.

    Args:
        text: Raw model output containing grounding tags.

    Returns:
        List of GroundingResult with parsed grounding data.
    """
    matches = re.findall(GROUNDING_PATTERN, text, re.DOTALL)
    results: list[GroundingResult] = []

    for _full_match, label, coords_str in matches:
        coordinates = _parse_coordinates(coords_str)
        if coordinates is not None:
            results.append(GroundingResult(label=label, coordinates=coordinates))

    return results
