"""Vehicle tracking in scan videos using Meta SAM 3.1."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from setu.meta_client import MetaClient, get_meta_client
from setu.video_io import iter_video_frames

logger = logging.getLogger(__name__)
TRACK_FPS = 10.0
MAX_HEIGHT = 480
VEHICLE_TYPES = ("truck", "bus", "car")
_FRAME_RECORD = re.compile(r"<(\d+)f(?:;[^>]*)?>(.*?)(?=<(?:\d+)f(?:;[^>]*)?>|$)", re.DOTALL)
_OBJECT_BOX = re.compile(r"(?:^|,)\s*(\d+)\s*<\|box;([^|]+)\|>")


def _prepare_video(source: str | Path, target: str | Path) -> int:
    import av
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate or 0)
        frames = iter_video_frames(source)
        first = next(frames, None)
        if first is None:
            raise ValueError("Traffic video contained no readable frames")
        height, width = first.shape[:2]
        if source_fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Could not determine traffic video dimensions or frame rate")
        scale = min(1.0, MAX_HEIGHT / height)
        out_size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), TRACK_FPS, out_size)
        if not writer.isOpened():
            raise ValueError("MP4 video encoding is unavailable")
        interval = source_fps / TRACK_FPS
        next_frame = 0.0
        count = 0
        index = 0
        for index, frame in enumerate(chain([first], frames)):
            if index + 1e-9 >= next_frame:
                if (frame.shape[1], frame.shape[0]) != out_size:
                    frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
                writer.write(frame)
                count += 1
                next_frame += interval
            index += 1
        writer.release()
        if count == 0:
            raise ValueError("Traffic video contained no readable frames")
        return count
    except Exception:
        raise


def _output_text(response: Dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(parts)


def _events_from_response(response: Dict[str, Any], kind: str, fps: float) -> List[Dict[str, Any]]:
    visible: Dict[str, set[int]] = {}
    for match in _FRAME_RECORD.finditer(_output_text(response)):
        frame = int(match.group(1))
        for obj in _OBJECT_BOX.finditer(match.group(2)):
            visible.setdefault(obj.group(1), set()).add(frame)
    events = []
    for frames in visible.values():
        ordered = sorted(frames)
        if not ordered:
            continue
        start = previous = ordered[0]
        for frame in ordered[1:] + [None]:
            if frame is not None and frame <= previous + 1:
                previous = frame
                continue
            duration = (previous - start + 1) / fps
            if duration >= 0.5:
                events.append({"t_start": start / fps, "t_end": (previous + 1) / fps, "type": kind})
            if frame is not None:
                start = previous = frame
    return events


def _plot_displacement(displacement: Any, fps: Optional[float], events: List[Dict[str, Any]], out_png: str | Path) -> None:
    if isinstance(displacement, dict):
        values = np.asarray(displacement.get("displacement", displacement.get("values", [])), dtype=float)
        times = np.asarray(displacement.get("time", displacement.get("times", [])), dtype=float)
    else:
        values = np.asarray(displacement, dtype=float).reshape(-1)
        times = np.arange(values.size, dtype=float) / float(fps or TRACK_FPS)
    if not values.size:
        raise ValueError("displacement must contain at least one sample")
    if times.size != values.size:
        times = np.arange(values.size, dtype=float) / float(fps or TRACK_FPS)
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(times, values, color="#174a7e", linewidth=1.2)
    for event in events:
        ax.axvspan(event["t_start"], event["t_end"], color="#e5a823", alpha=0.22)
    ax.set(xlabel="Time (s)", ylabel="Displacement", title="Displacement and vehicle traffic")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def track_traffic(
    video_path: str | Path,
    displacement: Any = None,
    fps: Optional[float] = None,
    out_png: Optional[str | Path] = None,
    meta_client: Optional[MetaClient] = None,
) -> Dict[str, Any]:
    """Track trucks, buses, and cars; return only events lasting at least 0.5 s."""
    client = meta_client or get_meta_client()
    if os.environ.get("SETU_PUBLIC_DEMO", "").strip().lower() in ("true", "1", "yes"):
        return {"events": [], "reason": "Traffic tracking is unavailable in public demo mode."}
    if client.mode != "mock" and not client.api_key:
        return {"events": [], "reason": "Traffic tracking requires a Meta API key."}
    try:
        with tempfile.TemporaryDirectory(prefix="setu-traffic-") as temp_dir:
            small = Path(temp_dir) / "traffic.mp4"
            frame_count = _prepare_video(video_path, small)
            events: List[Dict[str, Any]] = []
            for kind in VEHICLE_TYPES:
                response = client.track_video(small, kind, model="sam-3.1", fps=TRACK_FPS, frame_count=frame_count)
                if response is None:
                    return {"events": [], "reason": "SAM video tracking returned no response (public demo or missing key)."}
                events.extend(_events_from_response(response, kind, TRACK_FPS))
        events.sort(key=lambda item: (item["t_start"], item["type"]))
        logger.info("SAM 3.1 traffic video estimate: %d frames x 3 prompts at $0.20/1,000 frames = $%.4f", frame_count, frame_count * 3 * 0.20 / 1000)
        plot_path = None
        if displacement is not None:
            plot_path = Path(out_png) if out_png is not None else Path(video_path).with_suffix(".traffic.png")
            _plot_displacement(displacement, fps, events, plot_path)
        return {"events": events, "response_plot": str(plot_path) if plot_path is not None else None}
    except Exception as exc:
        logger.warning("Traffic tracking failed: %s", exc)
        return {"events": [], "reason": str(exc)}
