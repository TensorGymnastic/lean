"""Tests for OCR output post-processing."""

from __future__ import annotations

from lean.extraction.ocr_postprocess import clean_ocr_output


def test_strips_det_annotations() -> None:
    raw = "<|det|>title [428, 136, 600, 171]<|/det|>COUNCIL FOR"
    assert clean_ocr_output(raw) == "COUNCIL FOR"


def test_strips_page_markers() -> None:
    raw = "<PAGE>Some text</PAGE>"
    assert clean_ocr_output(raw) == "Some text"


def test_strips_image_tokens() -> None:
    raw = "<image>Multi page parsing."
    assert clean_ocr_output(raw) == "Multi page parsing."


def test_strips_grounding() -> None:
    raw = "<|grounding|>Find the table."
    assert clean_ocr_output(raw) == "Find the table."


def test_strips_ref_annotations() -> None:
    raw = "<|ref|>image<|/ref|> described here"
    assert clean_ocr_output(raw) == "described here"


def test_collapses_multiple_newlines() -> None:
    raw = "Line 1\n\n\n\n\nLine 2"
    assert clean_ocr_output(raw) == "Line 1\n\nLine 2"


def test_strips_trailing_whitespace() -> None:
    raw = "Line 1   \nLine 2\t\n"
    result = clean_ocr_output(raw)
    assert "   " not in result
    assert "\t" not in result


def test_full_ocr_output_sample() -> None:
    raw = (
        "<PAGE>"
        "<|det|>title [428, 136, 600, 171]<|/det|>COUNCIL FOR\n"
        "<|det|>title [269, 171, 745, 224]<|/det|>SIX SIGMA CERTIFICATION\n"
        "<|det|>text [365, 776, 650, 795]<|/det|>TRAINING MANUAL\n"
        "<image>\n"
        "</PAGE>"
    )
    result = clean_ocr_output(raw)
    assert "COUNCIL FOR" in result
    assert "SIX SIGMA CERTIFICATION" in result
    assert "TRAINING MANUAL" in result
    assert "<|det|>" not in result
    assert "<PAGE>" not in result
    assert "<image>" not in result


def test_empty_input() -> None:
    assert clean_ocr_output("") == ""
