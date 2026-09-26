from PIL import Image, ImageDraw
from scripts.data_preparation.mine_detail_crops import (
    accept,
    candidate_boxes,
    intersection_over_union,
    score_crop,
)


def test_candidate_boxes_are_deterministic_and_in_bounds() -> None:
    first = candidate_boxes(900, 700, 256, 20, 9)
    assert first == candidate_boxes(900, 700, 256, 20, 9)
    assert len(first) == 20
    assert all(0 <= x0 < x1 <= 900 and 0 <= y0 < y1 <= 700 for x0, y0, x1, y1 in first)
    assert candidate_boxes(512, 512, 512, 20, 9) == [(0, 0, 512, 512)]


def test_detail_score_rejects_flat_image() -> None:
    flat = Image.new("RGB", (512, 512), "white")
    detailed = flat.copy()
    draw = ImageDraw.Draw(detailed)
    for y in range(20, 500, 24):
        draw.text((20, y), "Detailed text 0123456789", fill="black")
    flat_score = score_crop(flat, "text")
    detailed_score = score_crop(detailed, "text")
    assert detailed_score.score > flat_score.score
    assert not accept(flat_score, "text")
    assert accept(detailed_score, "text")


def test_intersection_over_union() -> None:
    assert intersection_over_union((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1
