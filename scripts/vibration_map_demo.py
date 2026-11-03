"""Generate a synthetic four-post video and its per-instance frequency map.

Usage:
    python scripts/vibration_map_demo.py [output.png]

The video and frequencies are synthetic examples, not field measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from setu.vibration_map import build_vibration_map

FPS = 60.0
DURATION_S = 10.0
FREQUENCIES_HZ = (3.0, 4.0, 5.0, 6.0)
POST_BOXES = ((55, 82, 56, 210), (190, 82, 56, 210), (325, 82, 56, 210), (460, 82, 56, 210))


def generate_synthetic_video(path: Path) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    """Write a textured synthetic scene and return known post/background masks."""
    width, height = 640, 360
    rng = np.random.default_rng(73)
    base = np.full((height, width, 3), (120, 126, 130), dtype=np.uint8)
    # Stable, codec-friendly corners provide repeatable background features for
    # the camera stabilization tracker.
    for grid_y in range(18, height, 36):
        for grid_x in range(18, width, 36):
            shade = int(90 + ((grid_x // 36 + grid_y // 36) % 2) * 55)
            cv2.rectangle(base, (grid_x, grid_y), (grid_x + 13, grid_y + 13),
                          (shade, shade, shade), -1)
            cv2.circle(base, (grid_x + 24, grid_y + 8), 3, (70, 75, 80), -1)
    noise = rng.normal(0, 1.5, base.shape).astype(np.float32)
    frame_bg = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    frame_bg[310:330, :] = (100, 120, 130)
    background_mask = np.ones((height, width), dtype=bool)
    masks: List[Tuple[str, np.ndarray]] = []
    for index, (x, y, w, h) in enumerate(POST_BOXES):
        mask = np.zeros((height, width), dtype=bool)
        mask[y : y + h, x : x + w] = True
        masks.append((f"post {index + 1}", mask))
        background_mask[y - 18 : y + h + 18, x - 18 : x + w + 18] = False

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic video at {path}")
    textures = []
    for i, (_, _, w, h) in enumerate(POST_BOXES):
        texture_rng = np.random.default_rng(800 + i)
        texture = texture_rng.normal(135, 32, (h, w, 3)).clip(50, 220).astype(np.uint8)
        for stripe_x in range(4, w, 10):
            cv2.line(texture, (stripe_x, 0), (stripe_x, h - 1), (65, 70, 73), 1)
        cv2.rectangle(texture, (0, 0), (w - 1, h - 1), (45, 48, 50), 2)
        textures.append(texture)

    try:
        for frame_index in range(round(FPS * DURATION_S)):
            t = frame_index / FPS
            frame = frame_bg.copy()
            for i, (x, y, w, h) in enumerate(POST_BOXES):
                offset = 2.2 * np.sin(2 * np.pi * FREQUENCIES_HZ[i] * t)
                transform = np.float32([[1, 0, 0], [0, 1, offset]])
                moved = cv2.warpAffine(textures[i], transform, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=(122, 122, 122))
                alpha = cv2.warpAffine(np.full((h, w), 255, dtype=np.uint8), transform, (w, h),
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                roi = frame[y : y + h, x : x + w]
                a = alpha.astype(np.float32)[..., None] / 255.0
                frame[y : y + h, x : x + w] = (roi * (1 - a) + moved * a).astype(np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return masks, background_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="vibration_map_demo.png", help="Output PNG path")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    video_path = output.with_name(output.stem + "_synthetic.mp4")
    masks, background = generate_synthetic_video(video_path)
    records = build_vibration_map(video_path, masks, background, output)
    print("Synthetic instance frequency recovery")
    print(f"{'Instance':<12} {'Expected (Hz)':>14} {'Measured (Hz)':>15} {'Verdict':>10}")
    for index, record in enumerate(records):
        measured = record["frequency_hz"]
        measured_text = "n/a" if measured is None else f"{measured:.3f}"
        print(f"{record['label']:<12} {FREQUENCIES_HZ[index]:>14.2f} {measured_text:>15} {record['verdict']:>10}")
    print(f"PNG: {output.resolve()}")
    print(f"Synthetic video: {video_path.resolve()}")


if __name__ == "__main__":
    main()
