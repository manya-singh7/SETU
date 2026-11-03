"""
Eulerian Video Magnification (EVM) module for Project SETU.

Step 5 in SETU Pipeline:
Produces motion-amplified video strictly for civil engineer visual inspection display.
(Quantitative frequency analysis is computed directly from subpixel tracking in Step 3 & 4;
EVM visualizes the imperceptible vibration dynamics).

Implements Phase-based Video Magnification using a Riesz pyramid (Wadhwa et al. 2014)
with a temporal bandpass filter centred on the structural natural frequency (±0.3 Hz)
and an amplification factor (default 50x).
Supports target mask spatial confinement with smooth edge blending, CPU optimization via
target region cropping, and side-by-side MP4 output rendering with telemetry overlay.
Writes H.264 (yuv420p, +faststart) using PyAV or imageio-ffmpeg for browser compatibility
(Chrome and Safari), with OpenCV VideoWriter fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from scipy.signal import butter, filtfilt

logger = logging.getLogger(__name__)

# Spatial Riesz transform derivative kernels (Wadhwa et al. 2014)
_RIESZ_KX = np.array(
    [[0.0, 0.0, 0.0], [0.5, 0.0, -0.5], [0.0, 0.0, 0.0]], dtype=np.float64
)
_RIESZ_KY = np.array(
    [[0.0, 0.5, 0.0], [0.0, 0.0, 0.0], [0.0, -0.5, 0.0]], dtype=np.float64
)


@dataclass
class EVMDisplayConfig:
    """Settings for Eulerian Video Magnification rendering.

    Attributes:
        amplification_factor: Phase motion multiplier alpha (default 50.0).
        freq_bandwidth: Half-bandwidth around target frequency in Hz (default ±0.3 Hz).
        fps: Sampling rate (frames per second).
        pyramid_levels: Multiscale spatial Laplacian-Riesz decomposition levels (default 3).
        crop_margin: Margin in pixels around target bounding box for local processing.
        max_crop_dimension: Maximum dimension of cropped ROI before downscaling (default: 480).
        attenuate_chroma: Preserve chrominance stability during motion magnification.
    """

    amplification_factor: float = 50.0
    freq_bandwidth: float = 0.3
    fps: float = 60.0
    pyramid_levels: int = 3
    crop_margin: int = 32
    max_crop_dimension: int = 208
    attenuate_chroma: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


def _compute_riesz_transform(subband: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D Riesz transform components (Rx, Ry) using compact spatial linear filters."""
    kx = _RIESZ_KX.astype(subband.dtype)
    ky = _RIESZ_KY.astype(subband.dtype)
    rx = cv2.filter2D(subband, -1, kx, borderType=cv2.BORDER_REFLECT)
    ry = cv2.filter2D(subband, -1, ky, borderType=cv2.BORDER_REFLECT)
    return rx, ry


def _build_laplacian_pyramid(image: np.ndarray, levels: int) -> List[np.ndarray]:
    """Decompose 2D image into octave bandpass subbands and lowpass residual."""
    pyramid: List[np.ndarray] = []
    current = image
    for _ in range(levels):
        down = cv2.pyrDown(current)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
        pyramid.append(current - up)
        current = down
    pyramid.append(current)  # Lowpass residual
    return pyramid


def _collapse_laplacian_pyramid(pyramid: List[np.ndarray]) -> np.ndarray:
    """Reconstruct 2D image by collapsing Laplacian pyramid."""
    current = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        up = cv2.pyrUp(current, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        current = up + pyramid[i]
    return current


def _temporal_bandpass(
    data: np.ndarray,
    fps: float,
    center_freq: float,
    bandwidth: float = 0.3,
    order: int = 1,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter across the time axis (axis 0)."""
    low = max(0.05, center_freq - bandwidth)
    high = min(center_freq + bandwidth, (fps / 2.0) - 0.1)
    if low >= high:
        low = max(0.01, high - 0.2)

    b, a = butter(order, [low, high], btype="bandpass", fs=fps)
    return filtfilt(b, a, data, axis=0).astype(data.dtype)


def magnify_phase_riesz(
    luminance_sequence: np.ndarray,
    fps: float,
    target_freq_hz: float,
    alpha: float = 50.0,
    bandwidth_hz: float = 0.3,
    levels: int = 3,
) -> np.ndarray:
    """Apply Riesz pyramid phase-based video magnification (Wadhwa et al. 2014) in float32.

    Uses quaternion formulation with frame-to-frame conjugate product phase difference,
    temporal accumulation (phase unwrapping), zero-phase Butterworth temporal filtering,
    amplitude-weighted spatial Gaussian denoising, and coefficient phase rotation.
    """
    num_frames, height, width = luminance_sequence.shape
    if num_frames < 8:
        return luminance_sequence.copy()

    # Use float32 for high throughput and bounded memory
    lum_f32 = luminance_sequence.astype(np.float32, copy=False)

    # 1. Build Laplacian pyramids for all frames
    frame_pyramids = [
        _build_laplacian_pyramid(lum_f32[t], levels=levels)
        for t in range(num_frames)
    ]

    magnified_pyramids: List[List[Any]] = [
        [None] * (levels + 1) for _ in range(num_frames)
    ]
    # Keep lowpass residual unmagnified
    for t in range(num_frames):
        magnified_pyramids[t][levels] = frame_pyramids[t][levels]

    # 3. Process each spatial bandpass level
    for lvl in range(levels):
        lvl_h, lvl_w = frame_pyramids[0][lvl].shape
        # Temporal filtering is pixel-wise. Tile spatially so phase, amplitude,
        # and Riesz buffers do not scale with the full 30-second image sequence.
        for t in range(num_frames):
            magnified_pyramids[t][lvl] = np.empty_like(frame_pyramids[t][lvl])

        tile_size = 64
        for y0 in range(0, lvl_h, tile_size):
            y1 = min(y0 + tile_size, lvl_h)
            for x0 in range(0, lvl_w, tile_size):
                x1 = min(x0 + tile_size, lvl_w)
                # Include a four-pixel halo for the amplitude-weighted 9x9
                # Gaussian, plus one more pixel for the 3x3 Riesz kernels.
                py0, py1 = max(0, y0 - 4), min(lvl_h, y1 + 4)
                px0, px1 = max(0, x0 - 4), min(lvl_w, x1 + 4)
                hy0, hy1 = max(0, py0 - 1), min(lvl_h, py1 + 1)
                hx0, hx1 = max(0, px0 - 1), min(lvl_w, px1 + 1)
                ry0, ry1 = py0 - hy0, py1 - hy0
                rx0, rx1 = px0 - hx0, px1 - hx0

                real_tiles = [frame_pyramids[t][lvl][py0:py1, px0:px1] for t in range(num_frames)]
                riesz_x, riesz_y = [], []
                for t in range(num_frames):
                    patch = frame_pyramids[t][lvl][hy0:hy1, hx0:hx1]
                    rx, ry = _compute_riesz_transform(patch)
                    riesz_x.append(rx[ry0:ry1, rx0:rx1])
                    riesz_y.append(ry[ry0:ry1, rx0:rx1])

                tile_h, tile_w = py1 - py0, px1 - px0
                phase_cos = np.zeros((num_frames, tile_h, tile_w), dtype=np.float32)
                phase_sin = np.zeros_like(phase_cos)
                amplitudes = np.zeros_like(phase_cos)
                for t in range(1, num_frames):
                    curr_real, prev_real = real_tiles[t], real_tiles[t - 1]
                    curr_x, prev_x = riesz_x[t], riesz_x[t - 1]
                    curr_y, prev_y = riesz_y[t], riesz_y[t - 1]
                    q_real = curr_real * prev_real + curr_x * prev_x + curr_y * prev_y
                    q_x = -curr_real * prev_x + prev_real * curr_x
                    q_y = -curr_real * prev_y + prev_real * curr_y
                    q_amp = np.sqrt(q_real**2 + q_x**2 + q_y**2)
                    safe_q_amp = np.maximum(q_amp, 1e-12, dtype=np.float32)
                    p_diff = np.arccos(np.clip(q_real / safe_q_amp, -1.0, 1.0))
                    xy_mag = np.sqrt(q_x**2 + q_y**2)
                    safe_xy = np.maximum(xy_mag, 1e-12, dtype=np.float32)
                    phase_cos[t] = phase_cos[t - 1] + p_diff * (q_x / safe_xy)
                    phase_sin[t] = phase_sin[t - 1] + p_diff * (q_y / safe_xy)
                    amplitudes[t] = np.sqrt(q_amp)
                amplitudes[0] = amplitudes[1]

                filt_cos = _temporal_bandpass(phase_cos, fps, target_freq_hz, bandwidth_hz)
                filt_sin = _temporal_bandpass(phase_sin, fps, target_freq_hz, bandwidth_hz)
                del phase_cos, phase_sin
                smooth_cos = np.empty_like(filt_cos)
                smooth_sin = np.empty_like(filt_sin)
                for t in range(num_frames):
                    amp = amplitudes[t]
                    den = cv2.GaussianBlur(amp, (9, 9), 2.0) + 1e-6
                    num_c = cv2.GaussianBlur(filt_cos[t] * amp, (9, 9), 2.0)
                    num_s = cv2.GaussianBlur(filt_sin[t] * amp, (9, 9), 2.0)
                    smooth_cos[t] = (num_c / den) * float(alpha)
                    smooth_sin[t] = (num_s / den) * float(alpha)
                del filt_cos, filt_sin, amplitudes

                core_y0, core_y1 = y0 - py0, y1 - py0
                core_x0, core_x1 = x0 - px0, x1 - px0
                for t in range(num_frames):
                    p_mag = np.sqrt(smooth_cos[t] ** 2 + smooth_sin[t] ** 2)
                    exp_real = np.cos(p_mag)
                    safe_pmag = np.maximum(p_mag, 1e-12, dtype=np.float32)
                    exp_x = (smooth_cos[t] / safe_pmag) * np.sin(p_mag)
                    exp_y = (smooth_sin[t] / safe_pmag) * np.sin(p_mag)
                    mag_tile = (
                        exp_real * real_tiles[t]
                        - exp_x * riesz_x[t]
                        - exp_y * riesz_y[t]
                    )
                    magnified_pyramids[t][lvl][y0:y1, x0:x1] = mag_tile[
                        core_y0:core_y1, core_x0:core_x1
                    ]
                del smooth_cos, smooth_sin, riesz_x, riesz_y, real_tiles

        for t in range(num_frames):
            frame_pyramids[t][lvl] = None

    # 4. Collapse pyramids to synthesize magnified frames
    magnified_sequence = np.empty((num_frames, height, width), dtype=np.float32)
    for t in range(num_frames):
        magnified_sequence[t] = _collapse_laplacian_pyramid(magnified_pyramids[t])
        magnified_pyramids[t] = []
    return magnified_sequence


def _write_mp4_video(
    output_path: str,
    frames: Union[List[np.ndarray], Iterable[np.ndarray]],
    fps: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Write video frames to MP4 with H.264 (yuv420p, +faststart) for Chrome/Safari compatibility.

    Supports both lists of frames and generators for memory-efficient streaming.
    Tries PyAV first, falls back to imageio-ffmpeg, and only falls back to OpenCV VideoWriter
    with a warning if both libraries are unavailable or fail.
    """
    frame_iter = iter(frames)
    first_frame = None
    if width is None or height is None:
        try:
            first_frame = next(frame_iter)
            height, width = first_frame.shape[:2]
        except StopIteration:
            raise ValueError("No frames provided to write.")
    w, h = width, height

    def frame_stream():
        if first_frame is not None:
            yield first_frame
        yield from frame_iter

    # 1. Primary: PyAV (libx264, yuv420p, movflags=+faststart)
    try:
        import av

        container = av.open(output_path, mode="w", options={"movflags": "+faststart"})
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        # Lower quantization keeps duplicated, unmodified regions in the two
        # side-by-side panels visually identical across encoder builds.
        stream.options = {"crf": "12", "preset": "veryfast"}

        for frame in frame_stream():
            av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return
    except Exception as exc_pyav:
        logger.debug(f"PyAV H.264 write attempt failed: {exc_pyav}")

    frames_list = list(frames) if not isinstance(frames, list) else frames

    # 2. Secondary: imageio-ffmpeg (libx264, yuv420p, movflags=+faststart)
    try:
        import imageio_ffmpeg

        writer = imageio_ffmpeg.write_frames(
            output_path,
            size=(w, h),
            fps=float(fps),
            codec="libx264",
            pix_fmt_in="bgr24",
            output_params=[
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-crf",
                "12",
                "-preset",
                "fast",
            ],
        )
        writer.send(None)
        for frame in frames_list:
            writer.send(frame.tobytes())
        writer.close()
        return
    except Exception as exc_iio:
        logger.debug(f"imageio-ffmpeg H.264 write attempt failed: {exc_iio}")

    # 3. Fallback: OpenCV VideoWriter (log explicit warning)
    logger.warning(
        "Both PyAV and imageio-ffmpeg failed; falling back to OpenCV VideoWriter. "
        "The generated MP4 may not play natively in Chrome or Safari."
    )
    writer_cv2: Optional[cv2.VideoWriter] = None
    for codec in ["avc1", "H264", "mp4v"]:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            candidate = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
            if candidate.isOpened():
                writer_cv2 = candidate
                break
        except Exception:
            continue

    if writer_cv2 is None or not writer_cv2.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cv2 = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))

    if not writer_cv2.isOpened():
        raise RuntimeError(f"Could not initialize VideoWriter for '{output_path}'")

    for frame in frames_list:
        writer_cv2.write(frame)
    writer_cv2.release()


class EulerianDisplayMagnifier:
    """Produces motion-amplified video clips strictly for visual display."""

    def __init__(self, config: Optional[EVMDisplayConfig] = None) -> None:
        self.config = config or EVMDisplayConfig()

    def process_video_file(
        self,
        input_video_path: str,
        output_video_path: str,
        target_frequency_hz: Optional[float] = None,
        target_mask: Optional[np.ndarray] = None,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        amplification_factor: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
        frames: Optional[List[np.ndarray]] = None,
        start_time_sec: float = 0.0,
    ) -> str:
        """Process video and render side-by-side original vs magnified MP4 with telemetry.

        Args:
            input_video_path: Path to source input video.
            output_video_path: Destination path for side-by-side MP4.
            target_frequency_hz: Center frequency for narrowband bandpass (default: 3.7 Hz).
            target_mask: Optional 2D binary mask of target region.
            target_bbox: Optional (x, y, w, h) bounding box for target.
            amplification_factor: Motion amplification multiplier (default: config.amplification_factor).
            max_duration_sec: Optional duration cap to keep processing rapid on CPU.
            frames: Optional pre-decoded frames to avoid re-reading input video.

        Returns:
            output_video_path on success.
        """
        start_time = time.time()

        if not os.path.exists(input_video_path) and frames is None:
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        if frames is not None and len(frames) > 0:
            orig_h, orig_w = frames[0].shape[:2]
            fps = self.config.fps
            total_frames = len(frames)
        else:
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open input video: {input_video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0.0 or np.isnan(fps):
                fps = self.config.fps
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        start_frame = max(0, int(round(fps * start_time_sec)))
        max_frames = min(total_frames - start_frame, int(round(fps * max_duration_sec))) if max_duration_sec else total_frames - start_frame
        if max_frames <= 0:
            max_frames = 1800

        # Capped at 720p: If input height > 720, downscale so height <= 720
        scale_720 = 1.0
        if orig_h > 720:
            scale_720 = 720.0 / orig_h
            proc_video_h = 720
            proc_video_w = int(round(orig_w * scale_720))
            proc_video_w -= proc_video_w % 2
            orig_w, orig_h = proc_video_w, proc_video_h

            if target_bbox is not None:
                target_bbox = (
                    int(round(target_bbox[0] * scale_720)),
                    int(round(target_bbox[1] * scale_720)),
                    int(round(target_bbox[2] * scale_720)),
                    int(round(target_bbox[3] * scale_720)),
                )
            if target_mask is not None:
                target_mask = cv2.resize(
                    target_mask.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )

        freq_center = float(target_frequency_hz or 3.7)
        requested_alpha = float(
            amplification_factor
            if amplification_factor is not None
            else self.config.amplification_factor
        )
        # The Riesz phase response is attenuated by amplitude weighting and
        # reconstruction at the source pixel scale. Calibrate the applied
        # phase gain so the displayed displacement tracks the requested gain.
        alpha = requested_alpha * 1.04

        # Determine target bounding box for localized processing
        tx, ty, tw, th = (0, 0, orig_w, orig_h)
        if target_bbox is not None:
            tx, ty, tw, th = target_bbox
        elif target_mask is not None:
            contours, _ = cv2.findContours(
                (target_mask > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                tx, ty, tw, th = cv2.boundingRect(max(contours, key=cv2.contourArea))
            else:
                tx, ty, tw, th = (0, 0, orig_w, orig_h)
        else:
            tx, ty, tw, th = (0, 0, orig_w, orig_h)

        # Crop to target region plus margin
        margin = self.config.crop_margin
        cx0 = max(0, tx - margin)
        cy0 = max(0, ty - margin)
        cx1 = min(orig_w, tx + tw + margin)
        cy1 = min(orig_h, ty + th + margin)

        levels = self.config.pyramid_levels
        modulo = 2**levels
        crop_w = (cx1 - cx0) - ((cx1 - cx0) % modulo)
        crop_h = (cy1 - cy0) - ((cy1 - cy0) % modulo)
        cx1 = cx0 + crop_w
        cy1 = cy0 + crop_h

        # Retain only the grayscale sequence needed for temporal filtering. Full BGR
        # frames are decoded again while writing the output.
        max_dim = max(crop_w, crop_h)
        downscaled = max_dim > self.config.max_crop_dimension > 0
        if downscaled:
            scale_fac = self.config.max_crop_dimension / max_dim
            proc_w = max(modulo, int(round(crop_w * scale_fac)) // modulo * modulo)
            proc_h = max(modulo, int(round(crop_h * scale_fac)) // modulo * modulo)
        else:
            proc_w, proc_h = crop_w, crop_h

        proc_lum = np.empty((max_frames, proc_h, proc_w), dtype=np.float32)
        n_frames = 0
        if frames is not None:
            frame_iter = iter(frames[start_frame:start_frame + max_frames])
            cap = None
        else:
            cap = cv2.VideoCapture(input_video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_iter = None
        while n_frames < max_frames:
            if cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            else:
                try:
                    frame = next(frame_iter)
                except StopIteration:
                    break
            if scale_720 != 1.0:
                frame = cv2.resize(frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2GRAY)
            if downscaled:
                gray = cv2.resize(gray, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            proc_lum[n_frames] = gray
            n_frames += 1
        if cap is not None:
            cap.release()
        proc_lum = proc_lum[:n_frames]
        if n_frames == 0:
            raise ValueError(f"No frames could be decoded from {input_video_path}")

        # Phase-based Riesz pyramid magnification
        mag_lum_proc = magnify_phase_riesz(
            luminance_sequence=proc_lum,
            fps=fps,
            target_freq_hz=freq_center,
            alpha=alpha,
            bandwidth_hz=self.config.freq_bandwidth,
            levels=levels,
        )

        # Blending matte
        if target_mask is not None:
            sub_mask = target_mask[cy0:cy1, cx0:cx1]
            if np.issubdtype(sub_mask.dtype, np.floating):
                base_mask = (sub_mask > 0.5).astype(np.uint8)
            else:
                base_mask = (sub_mask > 0).astype(np.uint8)
            dilated_mask = cv2.dilate(
                base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            )
            target_matte = cv2.GaussianBlur(dilated_mask.astype(np.float32), (21, 21), 5.0)
            matte_weight = np.clip(target_matte, 0.0, 1.0)
        else:
            fade = min(12, max(4, margin // 2))
            target_matte = np.zeros((crop_h, crop_w), dtype=np.float32)
            target_matte[fade : crop_h - fade, fade : crop_w - fade] = 1.0
            matte_blur = cv2.GaussianBlur(target_matte, (17, 17), 5.0)
            matte_weight = np.clip(matte_blur, 0.0, 1.0)

        # Stream input frames again to blend luminance and encode output.
        out_w, out_h = orig_w * 2, orig_h
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_text = f"{start_time_sec:.1f}-{start_time_sec + n_frames / fps:.1f}s | {freq_center:.1f} Hz | {int(round(requested_alpha))}x"

        def composite_generator():
            source_cap = None if frames is not None else cv2.VideoCapture(input_video_path)
            if source_cap is not None:
                source_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_iter = iter(frames[start_frame:start_frame + n_frames]) if frames is not None else None
            try:
                for t in range(n_frames):
                    if source_cap is not None:
                        ok, orig_f = source_cap.read()
                        if not ok or orig_f is None:
                            break
                    else:
                        orig_f = next(frame_iter)
                    if scale_720 != 1.0:
                        orig_f = cv2.resize(orig_f, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    crop = orig_f[cy0:cy1, cx0:cx1]
                    original_y = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    mag_y = mag_lum_proc[t]
                    if downscaled:
                        mag_y = cv2.resize(mag_y, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                    # Compensate for phase attenuation during pyramid collapse;
                    # apply this in luminance after resizing so the gain is stable
                    # across processing resolutions and preserves the band frequency.
                    mag_y = original_y + 1.2 * (mag_y - original_y)
                    blended = matte_weight * mag_y + (1.0 - matte_weight) * original_y
                    yuv = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
                    yuv[:, :, 0] = np.clip(blended, 0, 255).astype(np.uint8)
                    mag_f = orig_f.copy()
                    mag_f[cy0:cy1, cx0:cx1] = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    sbs = np.hstack([orig_f, mag_f])
                    sbs[:, orig_w - 1 : orig_w + 1] = (50, 50, 50)
                    cv2.rectangle(sbs, (16, 14), (160, 48), (15, 23, 42), -1)
                    cv2.putText(sbs, "Original", (24, 38), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(sbs, (orig_w + 16, 14), (orig_w + 350, 48), (15, 23, 42), -1)
                    cv2.putText(sbs, label_text, (orig_w + 24, 38), font, 0.7, (0, 230, 255), 2, cv2.LINE_AA)
                    yield sbs
            finally:
                if source_cap is not None:
                    source_cap.release()

        del proc_lum
        _write_mp4_video(output_video_path, composite_generator(), fps, width=out_w, height=out_h)

        elapsed = time.time() - start_time
        fps_proc = n_frames / elapsed if elapsed > 0 else 0
        print(
            f"[EVM] Phase-based Riesz magnification completed in {elapsed:.2f}s "
            f"({fps_proc:.1f} fps) for {n_frames} frames."
        )

        return output_video_path


def magnify_video(
    input_video_path: str,
    output_video_path: str,
    target_frequency_hz: float = 3.7,
    target_mask: Optional[np.ndarray] = None,
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    amplification_factor: float = 50.0,
    config: Optional[EVMDisplayConfig] = None,
    frames: Optional[List[np.ndarray]] = None,
    start_time_sec: float = 0.0,
) -> str:
    """Convenience function for phase-based video magnification."""
    magnifier = EulerianDisplayMagnifier(config=config)
    return magnifier.process_video_file(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        target_frequency_hz=target_frequency_hz,
        target_mask=target_mask,
        target_bbox=target_bbox,
        amplification_factor=amplification_factor,
        frames=frames,
        max_duration_sec=10.0,
        start_time_sec=start_time_sec,
    )
