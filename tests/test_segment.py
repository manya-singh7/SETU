"""
Unit and integration tests for SETU structural segmentation module (setu.segment).

Tests cover:
- SegmentResult data shape compliance with setu.types (methods: "sam", "manual", "fallback")
- Strict enforcement: fallback results are NEVER labeled as "sam"
- Confidence verification: fallback/manual have confidence=None (no made-up scores)
- Fixed mask replication across all video frames (downstream motion stabilization requirement)
- Meta SAM 3.1 Promptable Concept Segmentation (PCS) via MetaClient (sam-3.1 model)
- Multi-instance matching: keeps all instances in metadata and selects largest/most central as target
- MetaClient SAM usage accounting ($2.50 per 1,000 images = $0.0025/image)
- MetaSAM31Segmenter raise_on_failure vs graceful fallback behavior
- Manual ROI segmentation with method="manual"
- Fallback offline segmenter determinism with method="fallback"
- Template matching displacement utility for QA drift checks
- Mask utility operations (bbox conversion, centroids, shifting, IoU, validation)
- Video loading from file, numpy arrays, and lists
- Factory functions and convenience entry points
- scripts/preview_masks.py CLI overlay execution
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import requests

from setu.meta_client import MetaClient
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
    compute_patch_ncc,
    compute_persistence,
    extract_bboxes_from_result,
    get_segmenter,
    load_video_frames,
    mask_to_bbox,
    parse_sam_grammar,
    segment_manual,
    segment_video,
    select_best_instance,
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
        assert isinstance(res.metadata, dict)
        assert res.metadata == {}


# ---------------------------------------------------------------------------
# Fallback Segmenter Tests
# ---------------------------------------------------------------------------

def test_fallback_segmenter_single_frame(synthetic_frame: np.ndarray):
    """Verify FallbackSegmenter produces deterministic masks, confidence=None, and method='fallback'."""
    segmenter = FallbackSegmenter()
    out = segmenter.segment(synthetic_frame)

    assert isinstance(out, SegmentationOutput)
    h, w = synthetic_frame.shape[:2]

    assert out.target_mask.shape == (h, w)
    assert out.background_mask.shape == (h, w)
    assert out.target_mask.dtype == bool
    assert out.background_mask.dtype == bool

    assert np.any(out.target_mask)
    assert np.any(out.background_mask)
    assert not np.any(np.logical_and(out.target_mask, out.background_mask))

    tx, ty, tw, th = out.target_bbox
    assert tw == int(w * 0.25)
    assert th == int(h * 0.25)

    # Strict confidence & method requirements
    assert out.confidence is None, "Fallback segmenter must not produce made-up confidence"
    assert out.method == "fallback", "Fallback segmenter must NEVER be labeled as 'sam'"


def test_fallback_segmenter_video_fixed_masks(synthetic_video_frames: list[np.ndarray]):
    """Verify FallbackSegmenter returns the same fixed masks for every frame with method='fallback'."""
    segmenter = FallbackSegmenter()
    res = segmenter.segment_video(synthetic_video_frames, fps=25.0)

    assert isinstance(res, SegmentResult)
    assert len(res.target_masks) == len(synthetic_video_frames)
    assert len(res.background_masks) == len(synthetic_video_frames)
    assert res.fps == 25.0
    assert res.method == "fallback"

    for i in range(len(synthetic_video_frames)):
        assert np.array_equal(res.target_masks[i], res.target_masks[0])
        assert np.array_equal(res.background_masks[i], res.background_masks[0])


# ---------------------------------------------------------------------------
# Manual Segmenter Tests
# ---------------------------------------------------------------------------

def test_manual_segmenter_roi(synthetic_frame: np.ndarray):
    """Verify ManualSegmenter strictly respects user ROIs, flags method='manual', and confidence=None."""
    h, w = synthetic_frame.shape[:2]
    t_box = (15, 20, 30, 40)
    b_box = (50, 80, 25, 20)

    seg = ManualSegmenter(target_bbox=t_box, background_bbox=b_box)
    out = seg.segment(synthetic_frame)

    assert out.target_bbox == t_box
    assert out.background_bbox == b_box
    assert out.method == "manual"
    assert out.confidence is None, "Manual segmenter must have confidence=None"

    assert compute_mask_area(out.target_mask) == 30 * 40
    assert compute_mask_area(out.background_mask) == 25 * 20

    res = seg.segment_video([synthetic_frame, synthetic_frame, synthetic_frame], fps=30.0)
    assert res.method == "manual"
    assert len(res.target_masks) == 3
    for i in range(3):
        assert np.array_equal(res.target_masks[i], res.target_masks[0])


# ---------------------------------------------------------------------------
# Meta SAM 3.1 & Strict Method Labeling Tests
# ---------------------------------------------------------------------------

def test_sam31_offline_never_labels_as_sam(synthetic_frame: np.ndarray):
    """Verify that when real SAM 3.1 is unreachable, method is 'fallback' and confidence is None."""
    client = MetaClient(mode="live", api_key="")  # Unconfigured live mode
    sam = MetaSAM31Segmenter(meta_client=client)

    out = sam.segment(synthetic_frame)
    assert out.method == "fallback"
    assert out.method != "sam"
    assert out.confidence is None, "Offline fallback must have confidence=None"

    res = sam.segment_video([synthetic_frame, synthetic_frame])
    assert res.method == "fallback"
    assert res.method != "sam"


def test_sam31_raise_on_failure(synthetic_frame: np.ndarray):
    """Verify MetaSAM31Segmenter raises RuntimeError when raise_on_failure=True and real SAM unreachable."""
    client = MetaClient(mode="live", api_key="")
    sam = MetaSAM31Segmenter(meta_client=client, raise_on_failure=True)
    with pytest.raises(RuntimeError, match="Real Meta SAM 3.1 model/API not reachable"):
        sam.segment(synthetic_frame)


def test_sam31_text_prompt_specialization(synthetic_frame: np.ndarray):
    """Verify Meta SAM 3.1 semantic heuristics tailor ROI based on structural keywords."""
    client = MetaClient(mode="live", api_key="")
    sam = MetaSAM31Segmenter(meta_client=client)

    cable_out = sam.segment(synthetic_frame, target_prompt="stay cable anchor", background_prompt="distant sky")
    assert cable_out.method == "fallback"
    assert cable_out.confidence is None

    deck_out = sam.segment(synthetic_frame, target_prompt="deck concrete girder", background_prompt="bedrock anchor")
    assert deck_out.target_bbox[2] > cable_out.target_bbox[2]

    pier_out = sam.segment(synthetic_frame, target_prompt="concrete pier column", background_prompt="bedrock anchor")
    assert pier_out.target_bbox[3] > deck_out.target_bbox[3]


def test_sam31_meta_client_mock_mode(synthetic_frame: np.ndarray):
    """Test MetaSAM31Segmenter integration with MetaClient in mock mode."""
    client = MetaClient(mode="mock")
    sam = MetaSAM31Segmenter(meta_client=client)
    out = sam.segment(synthetic_frame, target_prompt="concrete pillar")

    assert out.method == "sam"
    assert out.confidence == 0.94
    assert out.target_bbox is not None
    assert out.metadata["source"] == "meta_sam_3.1_api"
    assert out.metadata["mode"] == "mock"


def test_sam31_api_responses_grammar_parsing(synthetic_frame: np.ndarray):
    """Test MetaSAM31Segmenter correctly parses Meta Model API Responses token grammar."""
    raw_response = {
        "id": "resp_test_123",
        "object": "response",
        "model": "sam-3.1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "<0f>0<|box;x1=25;y1=30;x2=65;y2=80;w=160;h=120;score=0.96|><|mask;x=25;y=30;data=rle_test|>",
                    }
                ],
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = raw_response

    with patch("requests.post", return_value=mock_resp) as mock_post:
        client = MetaClient(api_key="meta_test_key_live", mode="live")
        sam = MetaSAM31Segmenter(meta_client=client)
        out = sam.segment(synthetic_frame)

        assert mock_post.called
        assert out.method == "sam"
        assert out.confidence == 0.96
        assert out.target_bbox == (25, 30, 40, 50)
        assert out.metadata["target_instance_count"] == 1


def test_sam31_multi_instance_selection_and_retention(synthetic_frame: np.ndarray):
    """Test that SAM 3.1 multiple matches are all retained in metadata, returning largest/central as target."""
    raw_response = {
        "id": "resp_multi_123",
        "object": "response",
        "model": "sam-3.1",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        # Instance 0: small corner box (x=10, y=10, w=15, h=15)
                        # Instance 1: large central pillar box (x=60, y=30, x2=100, y2=90 -> w=40, h=60)
                        "text": (
                            "<0f>0<|box;x1=10;y1=10;x2=25;y2=25;w=160;h=120;score=0.85|><|mask;data=m0|>,"
                            "1<|box;x1=60;y1=30;x2=100;y2=90;w=160;h=120;score=0.97|><|mask;data=m1|>"
                        ),
                    }
                ],
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = raw_response

    with patch("requests.post", return_value=mock_resp):
        client = MetaClient(api_key="meta_key_valid", mode="live")
        sam = MetaSAM31Segmenter(meta_client=client)
        out = sam.segment(synthetic_frame, target_prompt="concrete pillar")

        assert out.method == "sam"
        # All candidates must be retained in metadata
        assert "all_target_instances" in out.metadata
        assert len(out.metadata["all_target_instances"]) == 2
        assert out.metadata["target_instance_count"] == 2

        # Largest and most central instance (Instance 1) must be returned as target
        assert out.target_bbox == (60, 30, 40, 60)
        assert out.confidence == 0.97


def test_meta_client_sam_usage_accounting(tmp_path):
    """Verify that live SAM 3.1 calls log usage at $2.50 per 1,000 images ($0.0025 per call)."""
    usage_file = tmp_path / "usage.jsonl"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "output": [{"content": [{"type": "output_text", "text": "<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=10;h=10|>"}]}]
    }

    with patch("requests.post", return_value=mock_resp):
        client = MetaClient(
            api_key="meta_key_live",
            mode="live",
            usage_log_path=str(usage_file),
        )
        client.segment_image(image_base64="dummy_b64", prompt="pillar")

    assert usage_file.exists()
    with open(usage_file, "r", encoding="utf-8") as f:
        log_entry = json.loads(f.readline())
    assert log_entry["model"] == "sam-3.1"
    assert log_entry["images"] == 1
    assert log_entry["estimated_cost"] == 0.0025


def test_sam31_decoded_pillar_mask_is_not_rectangle():
    """Verify that decoded pillar mask from real SAM 3.1 fixture is NOT a rectangle."""
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sam_3_1_pillar_response.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        real_resp = json.load(f)

    # Decode with pillar frame dimensions (4000x6000)
    frame_shape = (4000, 6000)
    instances = parse_sam_grammar(real_resp, frame_shape)

    assert len(instances) == 1
    inst = instances[0]
    mask = inst["mask"]
    bbox = inst["bbox"]

    assert mask.shape == frame_shape
    bx, by, bw, bh = bbox
    bbox_area = bw * bh
    active_pixels = inst["area"]

    # Active pixels must be well below bbox area (not a simple filled rectangle)
    assert active_pixels > 0
    assert active_pixels < bbox_area * 0.75, (
        f"Pillar mask appears to be a filled rectangle: active={active_pixels}, bbox_area={bbox_area}"
    )
    # Real SAM response for this pillar contains no score
    assert inst["confidence"] is None


def test_sam31_no_background_returns_no_mask_and_provenance(synthetic_frame: np.ndarray):
    """Verify SAM returns no background mask and records provenance when background is not found."""
    target_fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sam_3_1_pillar_response.json")
    with open(target_fixture_path, "r", encoding="utf-8") as f:
        target_resp = json.load(f)

    empty_bg_resp = {
        "id": "resp_empty_bg",
        "object": "response",
        "model": "sam-3.1",
        "output": [{"content": [{"type": "output_text", "text": ""}]}],
    }

    mock_client = MagicMock()
    mock_client.mode = "mock"
    mock_client.segment_image.side_effect = [target_resp, empty_bg_resp]

    sam = MetaSAM31Segmenter(meta_client=mock_client)
    out = sam.segment(synthetic_frame, target_prompt="concrete pillar", background_prompt="distant bedrock")

    # Never substitute a hardcoded box
    assert out.background_mask is None
    assert out.background_bbox is None
    assert out.background_source == "none"
    assert out.target_source == "sam"
    assert out.target_mask is not None
    assert out.method == "sam"
    assert any("No stationary background found" in w for w in out.warnings)


def test_method_not_sam_if_mask_from_elsewhere(synthetic_frame: np.ndarray):
    """Verify that overall method must NOT say 'sam' if any mask came from fallback/manual."""
    fallback_seg = FallbackSegmenter()
    out = fallback_seg.segment(synthetic_frame)
    assert out.target_source == "fallback"
    assert out.background_source == "fallback"
    assert out.method == "fallback"
    assert out.method != "sam"

    manual_seg = ManualSegmenter(target_bbox=(10, 10, 20, 20), background_bbox=(30, 30, 20, 20))
    m_out = manual_seg.segment(synthetic_frame)
    assert m_out.target_source == "manual"
    assert m_out.background_source == "manual"
    assert m_out.method == "manual"
    assert m_out.method != "sam"


# ---------------------------------------------------------------------------
# Fixed-Masks Video Segmentation Tests
# ---------------------------------------------------------------------------

def test_fixed_masks_replicated_across_frames(synthetic_video_frames: list[np.ndarray]):
    """Verify segment_video returns the exact same fixed masks across all frames."""
    segmenter = get_segmenter(method="fallback")
    result = segmenter.segment_video(synthetic_video_frames, fps=30.0)

    n_frames = len(synthetic_video_frames)
    assert len(result.target_masks) == n_frames
    assert len(result.background_masks) == n_frames

    for i in range(1, n_frames):
        assert np.array_equal(result.target_masks[i], result.target_masks[0])
        assert np.array_equal(result.background_masks[i], result.background_masks[0])


def test_qa_template_displacement_utility():
    """Verify template matching displacement tracking accurately detects feature motion for QA."""
    h, w = 100, 100
    ref_gray = np.full((h, w), 50, dtype=np.uint8)
    ref_gray[40:50, 40:50] = 220

    curr_gray = np.full((h, w), 50, dtype=np.uint8)
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
    res_fallback = segment_video(synthetic_video_frames, fps=30.0, method="fallback")
    assert isinstance(res_fallback, SegmentResult)
    assert res_fallback.method == "fallback"
    assert len(res_fallback.target_masks) == len(synthetic_video_frames)

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
        assert "Confidence: None" in captured.out
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 1000


# ---------------------------------------------------------------------------
# Track Persistence & Failure Flagging Tests
# ---------------------------------------------------------------------------

def _create_textured_video(num_frames: int = 20, occlude_count: int = 0) -> list[np.ndarray]:
    """Helper creating synthetic video with textured pillar, optionally occluded for trailing frames."""
    h, w = 120, 160
    frames = []
    for i in range(num_frames):
        frame = np.full((h, w, 3), 190, dtype=np.uint8)
        # Textured pillar
        pillar_patch = np.zeros((60, 40, 3), dtype=np.uint8)
        for py in range(60):
            for px in range(40):
                pillar_patch[py, px] = [80 + (px % 6) * 5, 85 + (py % 5) * 4, 90 + ((px + py) % 7) * 3]
        frame[30:90, 60:100] = pillar_patch
        # Dark ground
        frame[95:120, :] = [45, 50, 40]

        # Apply occlusion to target region for the last `occlude_count` frames
        if i >= (num_frames - occlude_count):
            # Cover target with a black occlusion block
            cv2.rectangle(frame, (55, 25), (105, 95), (10, 10, 10), -1)

        frames.append(frame)
    return frames


def test_persistence_static_video_passes():
    """Verify that a static synthetic video passes track persistence with 100% valid frames."""
    frames = _create_textured_video(num_frames=20, occlude_count=0)
    segmenter = ManualSegmenter(target_bbox=(60, 30, 40, 60), background_bbox=(10, 95, 140, 25))
    res = segmenter.segment_video(frames, fps=30.0)

    assert res.metadata["status"] == "completed"
    assert res.metadata["failed"] is False
    assert res.metadata["is_valid"] is True
    assert "failure_reason" not in res.metadata

    tp = res.metadata["target_persistence"]
    assert tp is not None
    assert tp["passed"] is True
    assert tp["fraction_valid"] == 1.0
    assert tp["lowest_correlation"] >= 0.99
    assert tp["lowest_frame_idx"] == 0

    bp = res.metadata["background_persistence"]
    assert bp is not None
    assert bp["passed"] is True
    assert bp["fraction_valid"] == 1.0

    validation = validate_segment_result(res)
    assert validation["is_valid"] is True
    assert len(validation["errors"]) == 0


def test_persistence_occluded_video_fails():
    """Verify that target occlusion for 30% of frames fails persistence and marks segmentation failed."""
    # 20 frames, last 6 frames occluded (6/20 = 30% occlusion, 70% valid)
    frames = _create_textured_video(num_frames=20, occlude_count=6)
    segmenter = ManualSegmenter(target_bbox=(60, 30, 40, 60), background_bbox=(10, 95, 140, 25))
    res = segmenter.segment_video(frames, fps=30.0)

    # Must be marked failed
    assert res.metadata["status"] == "failed"
    assert res.metadata["failed"] is True
    assert res.metadata["is_valid"] is False
    assert "Target persistence failure" in res.metadata["failure_reason"]

    tp = res.metadata["target_persistence"]
    assert tp is not None
    assert tp["passed"] is False
    assert tp["fraction_valid"] == 0.70  # Exactly 14 / 20 = 70%
    assert tp["lowest_correlation"] < 0.50
    assert tp["lowest_frame_idx"] == 14
    assert tp["lowest_timestamp_sec"] == round(14 / 30.0, 3)

    validation = validate_segment_result(res)
    assert validation["is_valid"] is False
    assert any("Target persistence failure" in err for err in validation["errors"])


def test_persistence_raise_on_failure_raises_runtime_error():
    """Verify that segment_video raises RuntimeError when raise_on_failure=True and persistence fails."""
    frames = _create_textured_video(num_frames=20, occlude_count=6)
    segmenter = ManualSegmenter(target_bbox=(60, 30, 40, 60))
    segmenter.raise_on_failure = True

    with pytest.raises(RuntimeError, match="Target persistence failure"):
        segmenter.segment_video(frames, fps=30.0)


# ---------------------------------------------------------------------------
# Background Prompt Fallback Tests
# ---------------------------------------------------------------------------

def test_background_prompt_fallback_stops_at_first_match(synthetic_frame: np.ndarray):
    """Verify background prompt fallback tries candidates in order and stops at first match."""
    # Target prompt response
    target_resp = {
        "output": [{
            "content": [{
                "type": "output_text",
                "text": "<0f>0<|box;x1=60;y1=30;x2=100;y2=90;w=160;h=120;score=0.95|><|mask;x=60;y=30;data=rle_tgt|>",
            }]
        }]
    }
    # Initial background prompt ("sky") -> no instances
    empty_bg_resp = {
        "output": [{"content": [{"type": "output_text", "text": ""}]}]
    }
    # Fallback 1 ("building") -> matches 1 instance
    building_bg_resp = {
        "output": [{
            "content": [{
                "type": "output_text",
                "text": "<0f>0<|box;x1=10;y1=20;x2=50;y2=70;w=160;h=120;score=0.91|><|mask;x=10;y=20;data=rle_bld|>",
            }]
        }]
    }

    mock_client = MagicMock()
    mock_client.mode = "mock"
    # Target, primary bg (empty), fallback 1 (matched). Fallback 2 ("trees") should NOT be called.
    mock_client.segment_image.side_effect = [target_resp, empty_bg_resp, building_bg_resp]

    sam = MetaSAM31Segmenter(meta_client=mock_client)
    out = sam.segment(
        synthetic_frame,
        target_prompt="concrete column",
        background_prompt="sky reference",
        fallback_background_prompts=["building", "trees"],
    )

    # Primary failed, fallback 1 matched -> stops, so only 2 background attempts
    assert out.method == "sam"
    assert out.target_source == "sam"
    assert out.background_source == "sam"
    assert out.background_mask is not None
    assert out.background_bbox == (10, 20, 40, 50)
    assert out.metadata["matched_background_prompt"] == "building"

    attempts = out.metadata["background_prompts_attempted"]
    assert len(attempts) == 2
    assert attempts[0]["prompt"] == "sky reference"
    assert attempts[0]["matched"] is False
    assert attempts[0]["cost"] == 0.0025

    assert attempts[1]["prompt"] == "building"
    assert attempts[1]["matched"] is True
    assert attempts[1]["cost"] == 0.0025

    assert out.metadata["background_sam_cost"] == 0.0050
    # Assert client was called exactly 3 times (1 target + 2 background)
    assert mock_client.segment_image.call_count == 3


def test_background_prompt_fallback_exhaustion_never_hardcoded_box(synthetic_frame: np.ndarray):
    """Verify SAM returns no background mask and never substitutes a hardcoded box if all prompts fail."""
    target_resp = {
        "output": [{
            "content": [{
                "type": "output_text",
                "text": "<0f>0<|box;x1=60;y1=30;x2=100;y2=90;w=160;h=120;score=0.95|><|mask;x=60;y=30;data=rle_tgt|>",
            }]
        }]
    }
    empty_resp = {
        "output": [{"content": [{"type": "output_text", "text": ""}]}]
    }

    mock_client = MagicMock()
    mock_client.mode = "mock"
    # Target, primary bg (empty), fallback 1 (empty), fallback 2 (empty)
    mock_client.segment_image.side_effect = [target_resp, empty_resp, empty_resp, empty_resp]

    sam = MetaSAM31Segmenter(meta_client=mock_client)
    out = sam.segment(
        synthetic_frame,
        target_prompt="concrete column",
        background_prompt="bedrock anchor",
        fallback_background_prompts=["building", "ground"],
    )

    # Never substitute a hardcoded box
    assert out.background_mask is None
    assert out.background_bbox is None
    assert out.background_source == "none"
    assert out.target_source == "sam"
    assert out.metadata["matched_background_prompt"] is None

    attempts = out.metadata["background_prompts_attempted"]
    assert len(attempts) == 3
    assert all(not a["matched"] for a in attempts)
    assert out.metadata["background_sam_cost"] == 0.0075
    assert any("No stationary background found" in w for w in out.warnings)

