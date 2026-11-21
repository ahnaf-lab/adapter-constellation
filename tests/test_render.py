import pytest

from adapter_constellation.render import brightness_to_glyph, rasterize, render_frame


def star(x, y, brightness):
    return {"x": x, "y": y, "brightness": brightness}


def test_brightness_to_glyph_is_monotonic_and_bounds_to_space_and_densest_glyph():
    assert brightness_to_glyph(0.0) == " "
    glyphs = [brightness_to_glyph(b) for b in (0.0, 0.25, 0.5, 0.75, 1.0)]
    # Each step up in brightness should never pick an earlier (dimmer) glyph.
    from adapter_constellation.render import _GLYPHS

    indices = [_GLYPHS.index(g) for g in glyphs]
    assert indices == sorted(indices)
    assert glyphs[-1] == _GLYPHS[-1]


def test_brightness_to_glyph_rejects_out_of_range_input():
    with pytest.raises(ValueError):
        brightness_to_glyph(1.5)
    with pytest.raises(ValueError):
        brightness_to_glyph(-0.1)


def test_rasterize_places_center_and_corner_stars_in_expected_cells():
    stars = [star(0.0, 0.0, 1.0), star(-1.0, 1.0, 1.0), star(1.0, -1.0, 1.0)]
    grid = rasterize(stars, width=5, height=5)

    assert grid[2][2] != " "  # center of the disk maps to the center cell
    assert grid[0][0] != " "  # (-1, 1): leftmost column, topmost row
    assert grid[4][4] != " "  # (1, -1): rightmost column, bottommost row


def test_rasterize_keeps_the_brighter_star_when_two_share_a_cell():
    dim_first = rasterize([star(0.0, 0.0, 0.2), star(0.0, 0.0, 0.9)], width=3, height=3)
    bright_first = rasterize([star(0.0, 0.0, 0.9), star(0.0, 0.0, 0.2)], width=3, height=3)
    assert dim_first[1][1] == bright_first[1][1]


def test_render_frame_prepends_title_line_padded_to_width():
    lines = render_frame([star(0.0, 0.0, 1.0)], width=10, height=4, title="hi")
    assert len(lines) == 5
    assert lines[0] == "hi".ljust(10)
    assert all(len(line) == 10 for line in lines)


def test_render_frame_without_title_has_exactly_height_lines():
    lines = render_frame([], width=8, height=3)
    assert len(lines) == 3
    assert all(line == " " * 8 for line in lines)
