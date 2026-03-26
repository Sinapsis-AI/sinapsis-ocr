def denormalize_bbox(coords: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert normalized 0-999 coordinates to pixel coordinates.

    Args:
        coords: (x1, y1, x2, y2) in 0-999 range.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates.
    """
    x1, y1, x2, y2 = coords
    return (
        round(x1 * width / 999),
        round(y1 * height / 999),
        round(x2 * width / 999),
        round(y2 * height / 999),
    )
