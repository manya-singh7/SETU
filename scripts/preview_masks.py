#!/usr/bin/env python3
"""
Mask Preview Script for Project SETU.

Loads a civil infrastructure video recording (or generates a synthetic benchmark frame),
segments the structural target and stationary reference anchor on Frame 0,
overlays the masks and bounding boxes onto Frame 0, saves the resulting visualization
as a PNG file, and prints the segmentation method that produced them.

Usage:
    python scripts/preview_masks.py --video path/to/bridge.mp4 --target-prompt "corner of concrete pillar"
    python scripts/preview_masks.py --method fallback --output preview.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

# Ensure project root is in sys.path when running as a standalone script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from setu.segment import (
    compute_mask_area,
    get_segmenter,
    load_video_frames,
    mask_to_bbox,
)


def parse_bbox(bbox_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parse comma-separated 'x,y,w,h' string into an integer tuple."""
    if not bbox_str:
        return None
    parts = [int(p.strip()) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid bounding box '{bbox_str}'. Expected 'x,y,w,h'.")
    return (parts[0], parts[1], parts[2], parts[3])


def generate_synthetic_bridge_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate a synthetic 640x480 frame simulating a bridge pillar against ground."""
    frame = np.full((height, width, 3), 190, dtype=np.uint8)  # Sky / daylight background

    # Concrete Pier / Pillar in middle-ground
    px1, py1, pw, ph = int(width * 0.40), int(height * 0.25), int(width * 0.20), int(height * 0.55)
    cv2.rectangle(frame, (px1, py1), (px1 + pw, py1 + ph), (105, 105, 105), -1)
    # Add vertical joint texture lines
    cv2.line(frame, (px1 + pw // 3, py1), (px1 + pw // 3, py1 + ph), (80, 80, 80), 2)
    cv2.line(frame, (px1 + 2 * pw // 3, py1), (px1 + 2 * pw // 3, py1 + ph), (80, 80, 80), 2)

    # Bridge Deck (horizontal beam across top)
    cv2.rectangle(frame, (0, int(height * 0.20)), (width, int(height * 0.32)), (75, 75, 75), -1)

    # Stationary Bedrock / Embankment ground anchor
    cv2.rectangle(frame, (0, int(height * 0.78)), (width, height), (50, 60, 45), -1)

    return frame


def create_mask_overlay(
    frame: np.ndarray,
    target_mask: Optional[np.ndarray],
    background_mask: Optional[np.ndarray],
    target_bbox: Optional[Tuple[int, int, int, int]],
    background_bbox: Optional[Tuple[int, int, int, int]],
    target_prompt: str,
    background_prompt: str,
    method: str,
) -> np.ndarray:
    """Composite semi-transparent colored masks, bounding boxes, and label banners onto frame."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Target Mask: Emerald Green overlay
    target_color = np.array([30, 200, 30], dtype=np.uint8)
    # Background Reference Mask: Deep Sky Blue overlay
    background_color = np.array([230, 140, 20], dtype=np.uint8)

    alpha = 0.45

    if target_mask is not None and target_mask.any():
        canvas[target_mask] = (
            canvas[target_mask].astype(np.float32) * (1.0 - alpha) + target_color.astype(np.float32) * alpha
        ).astype(np.uint8)

    if background_mask is not None and background_mask.any():
        canvas[background_mask] = (
            canvas[background_mask].astype(np.float32) * (1.0 - alpha) + background_color.astype(np.float32) * alpha
        ).astype(np.uint8)

    # Draw Target Bounding Box and Annotation
    if target_bbox is not None:
        tx, ty, tw, th = target_bbox
        if tw > 0 and th > 0:
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                f"TARGET: {target_prompt}",
                (tx, max(20, ty - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Draw Background Reference Bounding Box and Annotation
    if background_bbox is not None:
        bx, by, bw, bh = background_bbox
        if bw > 0 and bh > 0:
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (255, 180, 0), 2)
            cv2.putText(
                canvas,
                f"BG REF: {background_prompt}",
                (bx, max(20, by - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

    # Informational Top Header Banner
    banner_height = 42
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)

    bg_info = "BG Anchor: Cyan" if background_mask is not None and background_mask.any() else "BG Anchor: None"
    method_tag = f"METHOD: {method.upper()}"
    cv2.putText(
        banner,
        f"SETU Frame 0 Segmentation  |  {method_tag}  |  (Target: Green, {bg_info})",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    combined = np.vstack([banner, canvas])
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overlay target and background segmentation masks on Frame 0 of bridge video."
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (.mp4, .mov, etc.). If omitted, uses synthetic bridge benchmark frame.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input still image (.jpg, .jpeg, .png). Takes precedence over --video.",
    )
    parser.add_argument(
        "--target-prompt",
        type=str,
        default="corner of concrete pillar",
        help="Open-vocabulary text cue for vibrating structural target.",
    )
    parser.add_argument(
        "--background-prompt",
        type=str,
        default="stationary ground reference",
        help="Text cue for stationary background anchor.",
    )
    parser.add_argument(
        "--fallback-background-prompts",
        type=str,
        default="building,ground,trees",
        help="Comma-separated fallback prompts for stationary background reference if primary prompt fails.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sam", "manual", "fallback"],
        default="sam",
        help="Segmentation strategy to run ('sam', 'manual', or 'fallback').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mask_preview.png",
        help="Destination path for the preview PNG overlay image.",
    )
    parser.add_argument(
        "--target-bbox",
        type=str,
        default=None,
        help="Manual target bounding box 'x,y,w,h' (used if method='manual').",
    )
    parser.add_argument(
        "--background-bbox",
        type=str,
        default=None,
        help="Manual background bounding box 'x,y,w,h' (used if method='manual').",
    )

    args = parser.parse_args()

    # Load image/video or fallback to synthetic frame
    media_path = args.image or args.video
    is_video = bool(args.video and not args.image)
    if media_path and os.path.exists(media_path):
        print(f"Loading input media: {media_path}")
        if is_video:
            frames, fps = load_video_frames(media_path)
            frame0 = frames[0]
        else:
            frames, fps = load_video_frames(media_path, max_frames=1)
            frame0 = frames[0]
    else:
        if media_path:
            print(f"Warning: Media file '{media_path}' not found. Generating synthetic bridge test frame.")
        else:
            print("No video or image specified. Using synthetic civil infrastructure benchmark frame.")
        frame0 = generate_synthetic_bridge_frame()
        frames = [frame0]
        fps = 30.0

    # Parse manual bboxes if supplied
    t_box = parse_bbox(args.target_bbox)
    b_box = parse_bbox(args.background_bbox)

    fallback_prompts = [p.strip() for p in args.fallback_background_prompts.split(",") if p.strip()]

    # Initialize segmenter
    segmenter = get_segmenter(
        method=args.method,
        target_bbox=t_box,
        background_bbox=b_box,
        fallback_background_prompts=fallback_prompts,
    )

    # Run segmentation
    output = segmenter.segment(
        frame=frame0,
        target_prompt=args.target_prompt,
        background_prompt=args.background_prompt,
        fallback_background_prompts=fallback_prompts,
    )

    # If multi-frame video, evaluate persistence across frames
    if is_video and len(frames) > 1:
        seg_res = segmenter.segment_video(
            video_source=frames,
            target_prompt=args.target_prompt,
            background_prompt=args.background_prompt,
            fps=fps,
            fallback_background_prompts=fallback_prompts,
        )
        output.metadata["target_persistence"] = seg_res.metadata.get("target_persistence")
        output.metadata["background_persistence"] = seg_res.metadata.get("background_persistence")
        output.metadata["status"] = seg_res.metadata.get("status")
        output.metadata["failed"] = seg_res.metadata.get("failed")
        output.metadata["failure_reason"] = seg_res.metadata.get("failure_reason")

    # Print which method and provenance produced the result
    print(f"Produced using method: {output.method}")
    conf_str = f"{output.confidence:.2f}" if output.confidence is not None else "None"
    print(f"  Confidence: {conf_str}")
    print(f"  Target Source: {output.target_source}")
    print(f"  Background Source: {output.background_source}")
    print(f"  Target BBox: {output.target_bbox} (Active pixels: {compute_mask_area(output.target_mask)})")
    if output.background_mask is not None:
        print(f"  Background BBox: {output.background_bbox} (Active pixels: {compute_mask_area(output.background_mask)})")
    else:
        print(f"  Background BBox: None (Active pixels: 0)")

    if "background_prompts_attempted" in output.metadata:
        attempts = output.metadata["background_prompts_attempted"]
        print(f"  Background Prompts Attempted ({len(attempts)}):")
        for att in attempts:
            status_desc = "matched" if att.get("matched") else "no match"
            print(f"    - Attempt #{att['attempt']}: '{att['prompt']}' -> {status_desc} (cost=${att.get('cost', 0.0):.4f})")
    if output.metadata.get("matched_background_prompt"):
        print(f"  Matched Background Prompt: '{output.metadata['matched_background_prompt']}'")
    if "background_sam_cost" in output.metadata:
        print(f"  Total Background SAM Cost: ${output.metadata['background_sam_cost']:.4f}")

    if "target_persistence" in output.metadata and output.metadata["target_persistence"]:
        tp = output.metadata["target_persistence"]
        print(f"  Target Persistence: valid={tp['fraction_valid']:.1%}, lowest_corr={tp['lowest_correlation']:.4f} "
              f"at frame {tp['lowest_frame_idx']} (t={tp['lowest_timestamp_sec']:.2f}s) | passed={tp['passed']}")
    if "background_persistence" in output.metadata and output.metadata["background_persistence"]:
        bp = output.metadata["background_persistence"]
        print(f"  Background Persistence: valid={bp['fraction_valid']:.1%}, lowest_corr={bp['lowest_correlation']:.4f} "
              f"at frame {bp['lowest_frame_idx']} (t={bp['lowest_timestamp_sec']:.2f}s) | passed={bp['passed']}")
    if output.metadata.get("failed"):
        print(f"  Segmentation FAILURE: {output.metadata.get('failure_reason')}")

    for w in output.warnings:
        print(f"  Warning: {w}")
    if "all_target_instances" in output.metadata and len(output.metadata["all_target_instances"]) > 1:
        print(f"  Total target candidates matched: {len(output.metadata['all_target_instances'])} (Selected largest/central)")
    if "fallback_reason" in output.metadata:
        print(f"  Fallback note: {output.metadata['fallback_reason']}")

    # Generate composite overlay
    overlay_img = create_mask_overlay(
        frame=frame0,
        target_mask=output.target_mask,
        background_mask=output.background_mask,
        target_bbox=output.target_bbox,
        background_bbox=output.background_bbox,
        target_prompt=args.target_prompt,
        background_prompt=output.background_prompt or args.background_prompt,
        method=output.method,
    )

    # Ensure parent output directory exists and write PNG
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    success = cv2.imwrite(args.output, overlay_img)
    if success:
        print(f"Saved mask overlay preview to: {os.path.abspath(args.output)}")
    else:
        print(f"Error: Failed to write image to {args.output}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
