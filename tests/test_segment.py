"""
Unit and integration tests for SETU structural segmentation module (setu.segment).

Tests cover:
- SegmentResult data shape compliance with setu.types (methods: "sam", "manual", "fallback")
- Strict enforcement: fallback results are NEVER labeled as "sam"
- Fixed mask replication across all video frames (downstream motion stabilization requirement)
- Meta SAM 3.1 Promptable Concept Segmentation (PCS) & mock API interactions
- MetaSAM31Segmenter raise_on_failure vs graceful fallback behavior
- Manual ROI segmentation with method="manual"
- Fallback offline segmenter determinism with method="fallback"
- Template matching displacement utility for QA drift checks
- Mask utility operations (bbox conversion, centroids, shifting, IoU, validation)
- Video loading from file, numpy arrays, and lists
- Factory functions and convenience entry points
- scripts/preview_masks.py CLI overlay execution
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import requests

from setu.segment import (
    BaseSegmenter,
    FallbackSegmenter,
    ManualSegmenter,
    MetaSAM31Segmenter,
    SegmentationOutput,
    bbox_to_mask,
    compute_mask_area,
    compute_mask_centroid,
    compute_mask_iou,
    extract_bboxes_from_result,
    get_segmenter,
    load_video_frames,
    mask_to_bbox,
    segment_manual,
    segment_video,
    shift_mask,
    track_roi_displacement,
    validate_segment_result,
)
from setu.types import SegmentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Create a synthetic 120x160 RGB frame representing bridge pillar against ground."""
    h, w = 120, 160
    frame = np.full((h, w, 3), 180, dtype=np.uint8)  # sky / background gray

    # Draw vertical concrete pillar in center
    frame[30:90, 60:100] = [80, 80, 80]

    # Draw dark bedrock ground at bottom
    frame[95:120, :] = [40, 40, 40]
    return frame


@pytest.fixture
def synthetic_video_frames(synthetic_frame: np.ndarray) -> list[np.ndarray]:
    """Create a 5-frame synthetic video sequence with slight simulated motion."""
    frames = []
    h, w = synthetic_frame.shape[:2]
    for i in range(5):
        f = synthetic_frame.copy()
        shift = i % 2
        f[30:90, (60 + shift) : (100 + shift)] = [80, 80, 80]
        frames.append(f)
    return frames


# ---------------------------------------------------------------------------
# Data Shape & Contract Tests
# ---------------------------------------------------------------------------

def test_segment_result_contract():
    """Verify SegmentResult matches the shared data shape required by setu.types."""
    h, w = 60, 80
    m1 = np.zeros((h, w), dtype=bool)
    m1[10:30, 20:40] = True
    bg1 = np.zeros((h, w), dtype=bool)
    bg1[40:55, 10:30] = True

    # Test with each valid Literal method
    for method in ("sam", "manual", "fallback"):
        res = SegmentResult(
            target_masks=[m1],
            background_masks=[bg1],
            fps=30.0,
            method=method,  # type: ignore[arg-type]
        )

        assert isinstance(res.target_masks, list)
        assert len(res.target_masks) == 1
        assert res.target_masks[0].shape == (h, w)
        assert res.target_masks[0].dtype == bool
        assert isinstance(res.background_masks, list)
        assert res.background_masks[0].shape == (h, w)
        assert res.background_masks[0].dtype == bool
        assert res.fps == 30.0
        assert res.method in ("sam", "manual", "fallback")


# ---------------------------------------------------------------------------
# Fallback Segmenter Tests
# ---------------------------------------------------------------------------

def test_fallback_segmenter_single_frame(synthetic_frame: np.ndarray):
    """Verify FallbackSegmenter produces deterministic masks and flags method='fallback'."""
    segmenter = FallbackSegmenter()
    out = segmenter.segment(synthetic_frame)

    assert isinstance(out, SegmentationOutput)
    h, w = synthetic_frame.shape[:2]

    # Check mask shapes & types
    assert out.target_mask.shape == (h, w)
    assert out.background_mask.shape == (h, w)
    assert out.target_mask.dtype == bool
    assert out.background_mask.dtype == bool

    # Check non-emptiness
    assert np.any(out.target_mask)
    assert np.any(out.background_mask)

    # Check non-overlap
    assert not np.any(np.logical_and(out.target_mask, out.background_mask))

    # Check bounding box consistency
    tx, ty, tw, th = out.target_bbox
    assert tw == int(w * 0.25)
    assert th == int(h * 0.25)
    assert out.confidence >= 0.9

    # Method MUST be "fallback", NEVER "sam"
    assert out.method == "fallback"


def test_fallback_segmenter_video_fixed_masks(synthetic_video_frames: list[np.ndarray]):
    """Verify FallbackSegmenter returns the same fixed masks for every frame with method='fallback'."""
    segmenter = FallbackSegmenter()
    res = segmenter.segment_video(synthetic_video_frames, fps=25.0)

    assert isinstance(res, SegmentResult)
    assert len(res.target_masks) == len(synthetic_video_frames)
    assert len(res.background_masks) == len(synthetic_video_frames)
    assert res.fps == 25.0
    assert res.method == "fallback"

    # All frames must have the exact same fixed masks (masks must NOT move per frame)
    for i in range(len(synthetic_video_frames)):
        assert np.array_equal(res.target_masks[i], res.target_masks[0])
        assert np.array_equal(res.background_masks[i], res.background_masks[0])


# ---------------------------------------------------------------------------
# Manual Segmenter Tests
# ---------------------------------------------------------------------------

def test_manual_segmenter_roi(synthetic_frame: np.ndarray):
    """Verify ManualSegmenter strictly respects user ROIs and flags method='manual'."""
    h, w = synthetic_frame.shape[:2]
    t_box = (15, 20, 30, 40)
    b_box = (50, 80, 25, 20)

    seg = ManualSegmenter(target_bbox=t_box, background_bbox=b_box)
    out = seg.segment(synthetic_frame)

    assert out.target_bbox == t_box
    assert out.background_bbox == b_box
    assert out.method == "manual"

    # Mask active area matches bbox area
    assert compute_mask_area(out.target_mask) == 30 * 40
    assert compute_mask_area(out.background_mask) == 25 * 20

    # Test video segmentation: method="manual" and fixed masks across frames
    res = seg.segment_video([synthetic_frame, synthetic_frame, synthetic_frame], fps=30.0)
    assert res.method == "manual"
    assert len(res.target_masks) == 3
    for i in range(3):
        assert np.array_equal(res.target_masks[i], res.target_masks[0])


# ---------------------------------------------------------------------------
# Meta SAM 3.1 & Strict Method Labeling Tests
# ---------------------------------------------------------------------------

def test_sam31_offline_never_labels_as_sam(synthetic_frame: np.ndarray, caplog: pytest.LogCaptureFixture):
    """Verify that when real SAM 3.1 is unreachable, method is 'fallback' (NEVER 'sam')."""
    sam = MetaSAM31Segmenter()  # Default offline without API key

    out = sam.segment(synthetic_frame)
    assert out.method == "fallback"
    assert out.method != "sam"

    # Verify video segmentation also sets method="fallback"
    res = sam.segment_video([synthetic_frame, synthetic_frame])
    assert res.method == "fallback"
    assert res.method != "sam"


def test_sam31_raise_on_failure(synthetic_frame: np.ndarray):
    """Verify MetaSAM31Segmenter raises RuntimeError when raise_on_failure=True and real SAM unreachable."""
    sam = MetaSAM31Segmenter(api_key=None, raise_on_failure=True)
    with pytest.raises(RuntimeError, match="Real Meta SAM 3.1 model/API not reachable"):
        sam.segment(synthetic_frame)


def test_sam31_text_prompt_specialization(synthetic_frame: np.ndarray):
    """Verify Meta SAM 3.1 semantic heuristics tailor ROI based on structural keywords."""
    sam = MetaSAM31Segmenter()

    cable_out = sam.segment(synthetic_frame, target_prompt="stay cable anchor", background_prompt="distant sky")
    assert cable_out.method == "fallback"

    deck_out = sam.segment(synthetic_frame, target_prompt="deck concrete girder", background_prompt="bedrock anchor")
    assert deck_out.target_bbox[2] > cable_out.target_bbox[2]  # deck width is broader

    pier_out = sam.segment(synthetic_frame, target_prompt="concrete pier column", background_prompt="bedrock anchor")
    assert pier_out.target_bbox[3] > deck_out.target_bbox[3]  # pier height is taller


def test_sam31_api_success_mock(synthetic_frame: np.ndarray):
    """Test MetaSAM31Segmenter labels method='sam' ONLY when Meta Model API actually succeeds."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "target_bbox": [25, 30, 40, 50],
        "background_bbox": [10, 95, 35, 20],
        "confidence": 0.98,
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        with patch("setu.config.settings.USE_LOCAL_FALLBACKS", False):
            sam = MetaSAM31Segmenter(api_key="meta_test_key_live", model_name="facebook/sam3.1")
            out = sam.segment(synthetic_frame)

            assert mock_post.called
            assert out.target_bbox == (25, 30, 40, 50)
            assert out.background_bbox == (10, 95, 35, 20)
            assert out.confidence == 0.98
            # Must be labeled "sam" because real API call succeeded
            assert out.method == "sam"
            assert out.metadata["source"] == "meta_sam_3.1_api"

            # Video result must also have method="sam"
            video_res = sam.segment_video([synthetic_frame, synthetic_frame])
            assert video_res.method == "sam"


def test_sam31_api_failure_graceful_fallback(synthetic_frame: np.ndarray):
    """Verify that when Meta SAM API call fails, it logs warning and labels method='fallback'."""
    with patch("requests.post", side_effect=requests.RequestException("Connection timeout")):
        with patch("setu.config.settings.USE_LOCAL_FALLBACKS", False):
            sam = MetaSAM31Segmenter(api_key="meta_test_key_abc", raise_on_failure=False)
            out = sam.segment(synthetic_frame)

            # Must fall back gracefully to heuristic and MUST be labeled "fallback"
            assert out is not None
            assert out.method == "fallback"
            assert out.method != "sam"


# ---------------------------------------------------------------------------
# Fixed-Masks Video Segmentation Tests
# ---------------------------------------------------------------------------

def test_fixed_masks_replicated_across_frames(synthetic_video_frames: list[np.ndarray]):
    """Verify segment_video returns the exact same fixed masks across all frames.

    Motion stabilization is handled downstream in setu/motion.py, so masks must not move.
    """
    segmenter = get_segmenter(method="fallback")
    result = segmenter.segment_video(synthetic_video_frames, fps=30.0)

    n_frames = len(synthetic_video_frames)
    assert len(result.target_masks) == n_frames
    assert len(result.background_masks) == n_frames

    # Check identical mask contents on all frames
    for i in range(1, n_frames):
        assert np.array_equal(result.target_masks[i], result.target_masks[0]), f"Frame {i} target mask shifted!"
        assert np.array_equal(result.background_masks[i], result.background_masks[0]), f"Frame {i} background mask shifted!"


def test_qa_template_displacement_utility():
    """Verify template matching displacement tracking accurately detects feature motion for QA."""
    h, w = 100, 100
    ref_gray = np.full((h, w), 50, dtype=np.uint8)
    ref_gray[40:50, 40:50] = 220

    curr_gray = np.full((h, w), 50, dtype=np.uint8)
    # Feature shifted right by 3 pixels, down by 2 pixels
    curr_gray[42:52, 43:53] = 220

    dx, dy = track_roi_displacement(ref_gray, curr_gray, bbox=(40, 40, 10, 10))
    assert dx == 3
    assert dy == 2


# ---------------------------------------------------------------------------
# Mask & Geometry Utility Tests
# ---------------------------------------------------------------------------

def test_bbox_and_mask_conversion():
    """Verify bbox_to_mask and mask_to_bbox bidirectionally."""
    shape = (100, 120)
    bbox = (20, 30, 25, 35)

    mask = bbox_to_mask(bbox, shape)
    recovered = mask_to_bbox(mask)

    assert recovered == bbox

    empty = np.zeros(shape, dtype=bool)
    assert mask_to_bbox(empty) == (0, 0, 0, 0)


def test_shift_mask_bounds():
    """Test shift_mask with positive, negative, and clipping offsets."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    initial_area = compute_mask_area(mask)

    shifted = shift_mask(mask, 3, 2)
    assert compute_mask_area(shifted) == initial_area
    assert shifted[7:12, 8:13].all()

    out_shifted = shift_mask(mask, 15, 0)
    assert compute_mask_area(out_shifted) < initial_area


def test_compute_mask_iou():
    """Verify IoU calculation on identical, overlapping, and disjoint masks."""
    m1 = np.zeros((10, 10), dtype=bool)
    m1[0:5, 0:5] = True

    m2 = np.zeros((10, 10), dtype=bool)
    m2[0:5, 0:5] = True
    assert compute_mask_iou(m1, m2) == 1.0

    m3 = np.zeros((10, 10), dtype=bool)
    m3[5:10, 5:10] = True
    assert compute_mask_iou(m1, m3) == 0.0

    m4 = np.zeros((10, 10), dtype=bool)
    m4[0:5, 2:7] = True
    iou = compute_mask_iou(m1, m4)
    assert 0.0 < iou < 1.0


def test_extract_bboxes_from_result():
    """Verify extracting downstream bounding boxes for motion tracking."""
    mask_t = np.zeros((50, 50), dtype=bool)
    mask_t[10:20, 10:20] = True

    mask_b = np.zeros((50, 50), dtype=bool)
    mask_b[30:40, 30:40] = True

    res = SegmentResult(target_masks=[mask_t], background_masks=[mask_b], fps=30.0, method="fallback")
    t_box, b_box = extract_bboxes_from_result(res, frame_idx=0)

    assert t_box == (10, 10, 10, 10)
    assert b_box == (30, 30, 10, 10)


def test_validate_segment_result():
    """Verify validation heuristics for valid and invalid SegmentResults."""
    m1 = np.zeros((30, 30), dtype=bool)
    m1[5:15, 5:15] = True
    bg1 = np.zeros((30, 30), dtype=bool)
    bg1[20:25, 20:25] = True

    valid_res = SegmentResult(target_masks=[m1], background_masks=[bg1], fps=30.0, method="fallback")
    report = validate_segment_result(valid_res)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0

    bad_res = SegmentResult(target_masks=[m1, m1], background_masks=[bg1], fps=30.0, method="fallback")
    bad_report = validate_segment_result(bad_res)
    assert bad_report["is_valid"] is False
    assert any("Mismatched" in e for e in bad_report["errors"])


# ---------------------------------------------------------------------------
# Video Loading & Source Formats Tests
# ---------------------------------------------------------------------------

def test_load_video_frames_from_arrays(synthetic_frame: np.ndarray):
    """Test loading video frames from 4D array, 3D array, and list."""
    arr4d = np.stack([synthetic_frame, synthetic_frame, synthetic_frame])
    frames, fps = load_video_frames(arr4d, default_fps=29.97)
    assert len(frames) == 3
    assert fps == 29.97

    gray = synthetic_frame[:, :, 0]
    arr3d = np.stack([gray, gray])
    frames_g, _ = load_video_frames(arr3d)
    assert len(frames_g) == 2

    frames_l, _ = load_video_frames([synthetic_frame, synthetic_frame])
    assert len(frames_l) == 2


def test_load_video_frames_from_file(synthetic_frame: np.ndarray):
    """Test writing and reading an actual video file through OpenCV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_bridge.mp4")
        h, w = synthetic_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

        for _ in range(5):
            writer.write(synthetic_frame)
        writer.release()

        frames, fps = load_video_frames(video_path)
        assert len(frames) == 5
        assert fps > 0.0


# ---------------------------------------------------------------------------
# Factory & Convenience Function Tests
# ---------------------------------------------------------------------------

def test_get_segmenter_factory():
    """Verify get_segmenter factory behavior with various options."""
    manual = get_segmenter("manual", target_bbox=(1, 2, 3, 4), background_bbox=(5, 6, 7, 8))
    assert isinstance(manual, ManualSegmenter)

    fallback = get_segmenter("fallback")
    assert isinstance(fallback, FallbackSegmenter)

    sam = get_segmenter("sam")
    assert isinstance(sam, MetaSAM31Segmenter)


def test_top_level_convenience_functions(synthetic_video_frames: list[np.ndarray]):
    """Verify segment_video and segment_manual helper functions."""
    res_sam = segment_video(synthetic_video_frames, fps=30.0, method="sam")
    assert isinstance(res_sam, SegmentResult)
    # Since offline, method must be "fallback", never "sam"
    assert res_sam.method == "fallback"
    assert len(res_sam.target_masks) == len(synthetic_video_frames)

    res_man = segment_manual(
        synthetic_video_frames,
        target_bbox=(20, 20, 30, 30),
        background_bbox=(70, 70, 20, 20),
        fps=30.0,
    )
    assert isinstance(res_man, SegmentResult)
    assert res_man.method == "manual"
    assert len(res_man.target_masks) == len(synthetic_video_frames)


# ---------------------------------------------------------------------------
# Preview Masks CLI Script Test
# ---------------------------------------------------------------------------

def test_preview_masks_script(capsys: pytest.CaptureFixture):
    """Verify scripts/preview_masks.py runs, outputs PNG, and prints method."""
    from scripts.preview_masks import main

    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = os.path.join(tmpdir, "preview.png")
        test_args = ["preview_masks.py", "--method", "fallback", "--output", out_png]

        with patch.object(sys, "argv", test_args):
            exit_code = main()
            assert exit_code == 0

        captured = capsys.readouterr()
        assert "Produced using method: fallback" in captured.out
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 1000
