"""Turns a list of constellation stars into a plain-text character frame.

`constellation.py` produces stars with an `(x, y)` position in the unit disk
and a `brightness` in `[0, 1]`; this module is the last, purely geometric
step before anything is drawn on screen: it buckets those positions onto a
fixed-size character grid and maps brightness onto a ramp of ASCII glyphs.
Nothing here touches curses or any other display library, so the mapping
itself -- which cell a star lands in, which glyph a brightness becomes -- is
fully testable without a terminal. `live.py` is the only module that takes
the resulting lines and actually draws them.
"""

from __future__ import annotations

Star = dict

# Dimmest to brightest. A space for brightness 0 means an unlit star leaves
# its cell blank rather than drawing a visible "off" marker.
_GLYPHS = " .:-=+*#@"


def brightness_to_glyph(brightness: float) -> str:
    """Maps a brightness in `[0, 1]` onto one glyph of a fixed dim-to-bright ramp."""
    if not (0.0 <= brightness <= 1.0):
        raise ValueError(f"brightness must be in [0, 1], got {brightness!r}")
    index = min(len(_GLYPHS) - 1, int(brightness * (len(_GLYPHS) - 1)))
    return _GLYPHS[index]


def _to_cell(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Maps a star's (x, y) in the unit disk to a (row, col) grid cell.

    `y` is flipped so that positive `y` (as `constellation.py` produces it)
    renders toward the top of the screen, matching how a reader expects a
    sky map to be oriented.
    """
    col = int((x + 1.0) / 2.0 * (width - 1))
    row = int((1.0 - y) / 2.0 * (height - 1))
    col = min(max(col, 0), width - 1)
    row = min(max(row, 0), height - 1)
    return row, col


def rasterize(stars: list[Star], width: int, height: int) -> list[list[str]]:
    """Projects `stars` onto a `height` x `width` grid of glyphs.

    When two stars land in the same cell, the brighter one wins -- a dim
    star never visually overwrites a brighter neighbor just by sharing a
    cell, regardless of which order `stars` lists them in.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    grid = [[" "] * width for _ in range(height)]
    brightest = [[-1.0] * width for _ in range(height)]
    for star in stars:
        row, col = _to_cell(star["x"], star["y"], width, height)
        if star["brightness"] > brightest[row][col]:
            brightest[row][col] = star["brightness"]
            grid[row][col] = brightness_to_glyph(star["brightness"])
    return grid


def render_frame(stars: list[Star], width: int, height: int, title: str | None = None) -> list[str]:
    """Renders `stars` to a list of plain-text lines, one per screen row.

    An optional `title` becomes its own line, padded/truncated to `width`,
    placed above the grid -- `live.py` uses it to show the current training
    step and loss alongside the constellation.
    """
    grid = rasterize(stars, width, height)
    lines = ["".join(row) for row in grid]
    if title is not None:
        lines = [title[:width].ljust(width)] + lines
    return lines
