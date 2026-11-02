"""
Unit and integration tests for SETU structural segmentation module (setu.segment).

Tests cover:
- SegmentResult data shape compliance with setu.types
- Meta SAM 3.1 Promptable Concept Segmentation (PCS) & mock API interactions
- Manual ROI segmentation and method="manual" flag
- Fallback offline segmenter determinism
- Object Multiplex video tracking across frame sequences
- Mask utility operations (bbox conversion, centroids, shifting, IoU, validation)
- Video loading from file, numpy arrays, and lists
- Factory functions and convenience entry points
"""

import os
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
    SegmentResult,
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
    validate_segment_result,
)


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
        # Add slight 1-pixel vibration shift to pillar across frames
        shift = i % 2
        f[30:90, (60 + shift) : (100 + shift)] = [80, 80, 80]
        frames.append(f)
    return frames


# ---------------------------------------------------------------------------
# Data Shape & Contract Tests
# ---------------------------------------------------------------------------

def test_segment_result_contract():
    """Verify SegmentResult matches the shared data shape required by SETU."""
    h, w = 60, 80
    m1 = np.zeros((h, w), dtype=bool)
    m1[10:30, 20:40] = True
    bg1 = np.zeros((h, w), dtype=bool)
    bg1[40:55, 10:30] = True

    res = SegmentResult(
        target_masks=[m1],
        background_masks=[bg1],
        fps=30.0,
        method="sam",
    )

    assert isinstance(res.target_masks, list)
    assert len(res.target_masks) == 1
    assert res.target_masks[0].shape == (h, w)
    assert res.target_masks[0].dtype == bool
    assert isinstance(res.background_masks, list)
    assert res.background_masks[0].shape == (h, w)
    assert res.background_masks[0].dtype == bool
    assert res.fps == 30.0
    assert res.method in ("sam", "manual")


# ---------------------------------------------------------------------------
# Fallback Segmenter Tests
# ---------------------------------------------------------------------------

def test_fallback_segmenter_single_frame(synthetic_frame: np.ndarray):
    """Verify FallbackSegmenter produces deterministic, valid masks and bounding boxes."""
    segmenter = FallbackSegmenter()
    out = segmenter.segment(synthetic_frame)

    assert isinstance(out, SegmentationOutput)
    h, w = synthetic_frame.shape[:2]

    # Check mask shapes
    assert out.target_mask.shape == (h, w)
    assert out.background_mask.shape == (h, w)
    assert out.target_mask.dtype == bool
    assert out.background_mask.dtype == bool

    # Check non-emptiness
    assert np.any(out.target_mask)
    assert np.any(out.background_mask)

    # Check that target and background do not overlap
    assert not np.any(np.logical_and(out.target_mask, out.background_mask))

    # Check bounding box consistency
    tx, ty, tw, th = out.target_bbox
    assert tw == int(w * 0.25)
    assert th == int(h * 0.25)
    assert out.confidence >= 0.9


def test_fallback_segmenter_video(synthetic_video_frames: list[np.ndarray]):
    """Verify FallbackSegmenter processes a video frame sequence into a SegmentResult."""
    segmenter = FallbackSegmenter()
    res = segmenter.segment_video(synthetic_video_frames, fps=25.0)

    assert isinstance(res, SegmentResult)
    assert len(res.target_masks) == len(synthetic_video_frames)
    assert len(res.background_masks) == len(synthetic_video_frames)
    assert res.fps == 25.0
    assert res.method == "sam"


# ---------------------------------------------------------------------------
# Manual Segmenter Tests
# ---------------------------------------------------------------------------

def test_manual_segmenter_roi(synthetic_frame: np.ndarray):
    """Verify ManualSegmenter strictly respects user-defined ROIs and flags method='manual'."""
    h, w = synthetic_frame.shape[:2]
    t_box = (15, 20, 30, 40)
    b_box = (50, 80, 25, 20)

    seg = ManualSegmenter(target_bbox=t_box, background_bbox=b_box)
    out = seg.segment(synthetic_frame)

    assert out.target_bbox == t_box
    assert out.background_bbox == b_box

    # Mask active area matches bbox area
    assert compute_mask_area(out.target_mask) == 30 * 40
    assert compute_mask_area(out.background_mask) == 25 * 20

    # Test video segmentation sets method="manual"
    res = seg.segment_video([synthetic_frame, synthetic_frame], fps=30.0)
    assert res.method == "manual"
    assert len(res.target_masks) == 2


# ---------------------------------------------------------------------------
# Meta SAM 3.1 & PCS Concept Prompt Tests
# ---------------------------------------------------------------------------

def test_sam31_text_prompt_specialization(synthetic_frame: np.ndarray):
    """Verify Meta SAM 3.1 semantic heuristics tailor ROI based on structural keywords."""
    sam = MetaSAM31Segmenter()

    # Cable prompt -> elevated target
    cable_out = sam.segment(synthetic_frame, target_prompt="stay cable anchor", background_prompt="distant sky")
    assert "sam3.1" in cable_out.metadata["mode"]

    # Deck prompt -> horizontal target
    deck_out = sam.segment(synthetic_frame, target_prompt="deck concrete girder", background_prompt="bedrock anchor")
    assert deck_out.target_bbox[2] > cable_out.target_bbox[2]  # deck width is broader

    # Pier prompt -> vertical target
    pier_out = sam.segment(synthetic_frame, target_prompt="concrete pier column", background_prompt="bedrock anchor")
    assert pier_out.target_bbox[3] > deck_out.target_bbox[3]  # pier height is taller


def test_sam31_api_success_mock(synthetic_frame: np.ndarray):
    """Test MetaSAM31Segmenter when Meta Model API returns successful bounding boxes."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "target_bbox": [25, 30, 40, 50],
        "background_bbox": [10, 95, 35, 20],
        "confidence": 0.98,
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        with patch("setu.config.settings.USE_LOCAL_FALLBACKS", False):
            sam = MetaSAM31Segmenter(api_key="meta_test_key_abc", model_name="facebook/sam3.1")
            out = sam.segment(synthetic_frame)

            assert mock_post.called
            assert out.target_bbox == (25, 30, 40, 50)
            assert out.background_bbox == (10, 95, 35, 20)
            assert out.confidence == 0.98
            assert out.metadata["source"] == "meta_sam_3.1_api"


def test_sam31_api_failure_graceful_fallback(synthetic_frame: np.ndarray):
    """Verify that MetaSAM31Segmenter falls back gracefully without crash when API fails."""
    with patch("requests.post", side_effect=requests.RequestException("Connection timeout")):
        with patch("setu.config.settings.USE_LOCAL_FALLBACKS", False):
            sam = MetaSAM31Segmenter(api_key="meta_test_key_abc")
            out = sam.segment(synthetic_frame)

            # Should fall back to local semantic heuristic instead of raising
            assert out is not None
            assert np.any(out.target_mask)
            assert np.any(out.background_mask)


# ---------------------------------------------------------------------------
# Object Multiplex Video Tracking Tests
# ---------------------------------------------------------------------------

def test_object_multiplex_video_tracking():
    """Verify that segment_video tracks translated structural features across frames."""
    h, w = 100, 100
    frames = []

    # Frame 0: High contrast patch at (40, 40)
    f0 = np.full((h, w), 50, dtype=np.uint8)
    f0[40:50, 40:50] = 220
    # Background anchor at (80, 10)
    f0[80:90, 10:20] = 220
    frames.append(f0)

    # Frame 1: Patch shifted right by 3 pixels to (43, 40)
    f1 = np.full((h, w), 50, dtype=np.uint8)
    f1[40:50, 43:53] = 220
    f1[80:90, 10:20] = 220
    frames.append(f1)

    seg = ManualSegmenter(target_bbox=(40, 40, 10, 10), background_bbox=(10, 80, 10, 10))
    res = seg.segment_video(frames, fps=30.0)

    assert len(res.target_masks) == 2
    c0 = compute_mask_centroid(res.target_masks[0])
    c1 = compute_mask_centroid(res.target_masks[1])

    assert c0 is not None and c1 is not None
    # Centroid x should have shifted right
    assert c1[0] >= c0[0]


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

    # Empty mask test
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

    # Shift partially out of frame
    out_shifted = shift_mask(mask, 15, 0)
    assert compute_mask_area(out_shifted) < initial_area


def test_compute_mask_iou():
    """Verify IoU calculation on identical, overlapping, and disjoint masks."""
    m1 = np.zeros((10, 10), dtype=bool)
    m1[0:5, 0:5] = True  # area 25

    m2 = np.zeros((10, 10), dtype=bool)
    m2[0:5, 0:5] = True

    assert compute_mask_iou(m1, m2) == 1.0

    m3 = np.zeros((10, 10), dtype=bool)
    m3[5:10, 5:10] = True
    assert compute_mask_iou(m1, m3) == 0.0

    # Partial overlap
    m4 = np.zeros((10, 10), dtype=bool)
    m4[0:5, 2:7] = True  # overlap 15
    iou = compute_mask_iou(m1, m4)
    assert 0.0 < iou < 1.0


def test_extract_bboxes_from_result():
    """Verify extracting downstream bounding boxes for motion tracking."""
    mask_t = np.zeros((50, 50), dtype=bool)
    mask_t[10:20, 10:20] = True

    mask_b = np.zeros((50, 50), dtype=bool)
    mask_b[30:40, 30:40] = True

    res = SegmentResult(target_masks=[mask_t], background_masks=[mask_b], fps=30.0)
    t_box, b_box = extract_bboxes_from_result(res, frame_idx=0)

    assert t_box == (10, 10, 10, 10)
    assert b_box == (30, 30, 10, 10)


def test_validate_segment_result():
    """Verify validation heuristics for valid and invalid SegmentResults."""
    m1 = np.zeros((30, 30), dtype=bool)
    m1[5:15, 5:15] = True
    bg1 = np.zeros((30, 30), dtype=bool)
    bg1[20:25, 20:25] = True

    valid_res = SegmentResult(target_masks=[m1], background_masks=[bg1], fps=30.0)
    report = validate_segment_result(valid_res)
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0

    # Mismatched frames
    bad_res = SegmentResult(target_masks=[m1, m1], background_masks=[bg1], fps=30.0)
    bad_report = validate_segment_result(bad_res)
    assert bad_report["is_valid"] is False
    assert any("Mismatched" in e for e in bad_report["errors"])

    # Negative FPS
    fps_bad = SegmentResult(target_masks=[m1], background_masks=[bg1], fps=-1.0)
    assert validate_segment_result(fps_bad)["is_valid"] is False


# ---------------------------------------------------------------------------
# Video Loading & Source Formats Tests
# ---------------------------------------------------------------------------

def test_load_video_frames_from_arrays(synthetic_frame: np.ndarray):
    """Test loading video frames from 4D array, 3D array, and list."""
    # 4D array (N, H, W, C)
    arr4d = np.stack([synthetic_frame, synthetic_frame, synthetic_frame])
    frames, fps = load_video_frames(arr4d, default_fps=29.97)
    assert len(frames) == 3
    assert fps == 29.97

    # 3D array (N, H, W)
    gray = synthetic_frame[:, :, 0]
    arr3d = np.stack([gray, gray])
    frames_g, _ = load_video_frames(arr3d)
    assert len(frames_g) == 2

    # List of frames
    frames_l, _ = load_video_frames([synthetic_frame, synthetic_frame])
    assert len(frames_l) == 2

    # Unsupported type
    with pytest.raises(TypeError):
        load_video_frames(12345)  # type: ignore[arg-type]


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

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_video_frames("non_existent_bridge_video.mp4")


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
    assert res_sam.method == "sam"
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
