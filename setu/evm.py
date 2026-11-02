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
from typing import Any, Dict, List, Optional, Tuple, Union

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
    max_crop_dimension: int = 480
    attenuate_chroma: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


def _compute_riesz_transform(subband: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D Riesz transform components (Rx, Ry) using compact spatial linear filters."""
    rx = cv2.filter2D(subband, -1, _RIESZ_KX, borderType=cv2.BORDER_REFLECT)
    ry = cv2.filter2D(subband, -1, _RIESZ_KY, borderType=cv2.BORDER_REFLECT)
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
    return filtfilt(b, a, data, axis=0)


def magnify_phase_riesz(
    luminance_sequence: np.ndarray,
    fps: float,
    target_freq_hz: float,
    alpha: float = 50.0,
    bandwidth_hz: float = 0.3,
    levels: int = 3,
) -> np.ndarray:
    """Apply Riesz pyramid phase-based video magnification (Wadhwa et al. 2014).

    Uses quaternion formulation with frame-to-frame conjugate product phase difference,
    temporal accumulation (phase unwrapping), zero-phase Butterworth temporal filtering,
    amplitude-weighted spatial Gaussian denoising, and coefficient phase rotation.

    Args:
        luminance_sequence: 3D array of grayscale / Y channel frames (T, H, W).
        fps: Video frame rate.
        target_freq_hz: Structural natural frequency to magnify.
        alpha: Motion amplification factor (default 50).
        bandwidth_hz: Half-bandwidth around target frequency (default 0.3 Hz).
        levels: Number of Laplacian spatial pyramid levels (default 3).

    Returns:
        3D array of magnified luminance frames (T, H, W).
    """
    num_frames, height, width = luminance_sequence.shape
    if num_frames < 8:
        return luminance_sequence.copy()

    # 1. Build Laplacian pyramids for all frames
    frame_pyramids = [
        _build_laplacian_pyramid(luminance_sequence[t], levels=levels)
        for t in range(num_frames)
    ]

    # 2. Compute spatial Riesz transform components (Rx, Ry) for all bandpass levels
    riesz_x = [[None] * levels for _ in range(num_frames)]
    riesz_y = [[None] * levels for _ in range(num_frames)]
    for t in range(num_frames):
        for lvl in range(levels):
            rx, ry = _compute_riesz_transform(frame_pyramids[t][lvl])
            riesz_x[t][lvl] = rx
            riesz_y[t][lvl] = ry

    magnified_pyramids: List[List[Any]] = [
        [None] * (levels + 1) for _ in range(num_frames)
    ]
    # Keep lowpass residual unmagnified
    for t in range(num_frames):
        magnified_pyramids[t][levels] = frame_pyramids[t][levels]

    # 3. Process each spatial bandpass level
    for lvl in range(levels):
        lvl_h, lvl_w = frame_pyramids[0][lvl].shape

        phase_cos = np.zeros((num_frames, lvl_h, lvl_w), dtype=np.float64)
        phase_sin = np.zeros((num_frames, lvl_h, lvl_w), dtype=np.float64)
        amplitudes = np.zeros((num_frames, lvl_h, lvl_w), dtype=np.float64)

        # Compute quaternionic phase difference between consecutive frames (t and t-1)
        # and unwrap phase by accumulation (Wadhwa et al. 2014 pseudocode)
        for t in range(1, num_frames):
            curr_real = frame_pyramids[t][lvl]
            curr_x = riesz_x[t][lvl]
            curr_y = riesz_y[t][lvl]

            prev_real = frame_pyramids[t - 1][lvl]
            prev_x = riesz_x[t - 1][lvl]
            prev_y = riesz_y[t - 1][lvl]

            # Quaternion conjugate product: q_curr * conjugate(q_prev)
            q_real = curr_real * prev_real + curr_x * prev_x + curr_y * prev_y
            q_x = -curr_real * prev_x + prev_real * curr_x
            q_y = -curr_real * prev_y + prev_real * curr_y

            q_amp = np.sqrt(q_real**2 + q_x**2 + q_y**2)
            safe_q_amp = np.maximum(q_amp, 1e-12)

            cos_arg = np.clip(q_real / safe_q_amp, -1.0, 1.0)
            p_diff = np.arccos(cos_arg)

            xy_mag = np.sqrt(q_x**2 + q_y**2)
            safe_xy = np.maximum(xy_mag, 1e-12)
            cos_theta = q_x / safe_xy
            sin_theta = q_y / safe_xy

            p_diff_cos = p_diff * cos_theta
            p_diff_sin = p_diff * sin_theta

            # Accumulate phase difference (temporal phase unwrapping)
            phase_cos[t] = phase_cos[t - 1] + p_diff_cos
            phase_sin[t] = phase_sin[t - 1] + p_diff_sin
            amplitudes[t] = np.sqrt(q_amp)

        amplitudes[0] = amplitudes[1]

        # Temporal bandpass filter around resonant frequency
        filt_cos = _temporal_bandpass(phase_cos, fps, target_freq_hz, bandwidth_hz)
        filt_sin = _temporal_bandpass(phase_sin, fps, target_freq_hz, bandwidth_hz)

        # Amplitude-weighted spatial Gaussian denoising (Wadhwa et al. Eq. 23)
        # Prevents phase-wrapping discontinuities and ringing artifacts
        smooth_cos = np.zeros_like(filt_cos)
        smooth_sin = np.zeros_like(filt_sin)
        ksize = 9
        sigma = 2.0
        for t in range(num_frames):
            amp = amplitudes[t]
            den = cv2.GaussianBlur(amp, (ksize, ksize), sigma) + 1e-6
            num_c = cv2.GaussianBlur(filt_cos[t] * amp, (ksize, ksize), sigma)
            num_s = cv2.GaussianBlur(filt_sin[t] * amp, (ksize, ksize), sigma)
            smooth_cos[t] = num_c / den
            smooth_sin[t] = num_s / den

        # Magnify phase variations
        mag_phase_cos = alpha * smooth_cos
        mag_phase_sin = alpha * smooth_sin

        # Reconstruct phase-shifted coefficient real part (Wadhwa et al. Eq. 11 & 12)
        for t in range(num_frames):
            p_mag = np.sqrt(mag_phase_cos[t] ** 2 + mag_phase_sin[t] ** 2)
            exp_real = np.cos(p_mag)
            safe_pmag = np.maximum(p_mag, 1e-12)
            exp_x = (mag_phase_cos[t] / safe_pmag) * np.sin(p_mag)
            exp_y = (mag_phase_sin[t] / safe_pmag) * np.sin(p_mag)

            mag_res = (
                exp_real * frame_pyramids[t][lvl]
                - exp_x * riesz_x[t][lvl]
                - exp_y * riesz_y[t][lvl]
            )
            magnified_pyramids[t][lvl] = mag_res

    # 4. Collapse pyramids to synthesize magnified frames
    magnified_sequence = np.array(
        [_collapse_laplacian_pyramid(magnified_pyramids[t]) for t in range(num_frames)],
        dtype=np.float64,
    )
    return magnified_sequence


def _write_mp4_video(
    output_path: str,
    frames: List[np.ndarray],
    fps: float,
) -> None:
    """Write video frames to MP4 with H.264 (yuv420p, +faststart) for Chrome/Safari compatibility.

    Tries PyAV first, falls back to imageio-ffmpeg, and only falls back to OpenCV VideoWriter
    with a warning if both libraries are unavailable or fail.
    """
    if not frames:
        raise ValueError("No frames provided to write.")

    h, w = frames[0].shape[:2]

    # 1. Primary: PyAV (libx264, yuv420p, movflags=+faststart)
    try:
        import av

        container = av.open(output_path, mode="w", options={"movflags": "+faststart"})
        stream = container.add_stream("libx264", rate=int(round(fps)))
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}

        for frame in frames:
            av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return
    except Exception as exc_pyav:
        logger.debug(f"PyAV H.264 write attempt failed: {exc_pyav}")

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
                "18",
                "-preset",
                "fast",
            ],
        )
        writer.send(None)
        for frame in frames:
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

    for frame in frames:
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

        Returns:
            output_video_path on success.
        """
        start_time = time.time()

        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0.0 or np.isnan(fps):
            fps = self.config.fps

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames = (
            int(round(fps * max_duration_sec)) if max_duration_sec else total_frames
        )
        if max_frames <= 0:
            max_frames = 1800

        # Read frames
        raw_frames: List[np.ndarray] = []
        while len(raw_frames) < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            raw_frames.append(frame)
        cap.release()

        n_frames = len(raw_frames)
        if n_frames == 0:
            raise ValueError(f"No frames could be decoded from {input_video_path}")

        freq_center = float(target_frequency_hz or 3.7)
        alpha = float(
            amplification_factor
            if amplification_factor is not None
            else self.config.amplification_factor
        )

        # Determine target bounding box for localized CPU processing
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

        # Snap dimensions to divisible by 2^levels (e.g. 8 for 3 levels)
        levels = self.config.pyramid_levels
        modulo = 2**levels
        crop_w = (cx1 - cx0) - ((cx1 - cx0) % modulo)
        crop_h = (cy1 - cy0) - ((cy1 - cy0) % modulo)
        cx1 = cx0 + crop_w
        cy1 = cy0 + crop_h

        # Extract crops across all frames
        crops = np.array([raw_frames[t][cy0:cy1, cx0:cx1] for t in range(n_frames)])

        # Convert to YUV for separate luminance processing (prevents color artifacts)
        yuv_crops = np.array(
            [cv2.cvtColor(c, cv2.COLOR_BGR2YUV) for c in crops], dtype=np.float64
        )
        lum_seq = yuv_crops[:, :, :, 0]

        # Downscale crop if it exceeds max_crop_dimension to keep CPU runtime minimal
        max_dim = max(crop_w, crop_h)
        downscaled = False
        if max_dim > self.config.max_crop_dimension and self.config.max_crop_dimension > 0:
            scale_fac = self.config.max_crop_dimension / max_dim
            proc_w = int(round(crop_w * scale_fac))
            proc_h = int(round(crop_h * scale_fac))
            proc_w -= proc_w % modulo
            proc_h -= proc_h % modulo
            proc_lum = np.array(
                [
                    cv2.resize(lum_seq[t], (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                    for t in range(n_frames)
                ]
            )
            downscaled = True
        else:
            proc_lum = lum_seq
            proc_w = crop_w
            proc_h = crop_h

        # Apply Phase-based Riesz Pyramid Magnification
        mag_lum_proc = magnify_phase_riesz(
            luminance_sequence=proc_lum,
            fps=fps,
            target_freq_hz=freq_center,
            alpha=alpha,
            bandwidth_hz=self.config.freq_bandwidth,
            levels=levels,
        )

        if downscaled:
            mag_lum = np.array(
                [
                    cv2.resize(mag_lum_proc[t], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                    for t in range(n_frames)
                ]
            )
        else:
            mag_lum = mag_lum_proc

        # Spatial blending mask: magnify strictly inside target, smooth transition at borders
        if target_mask is not None:
            sub_mask = target_mask[cy0:cy1, cx0:cx1]
            if np.issubdtype(sub_mask.dtype, np.floating):
                base_mask = (sub_mask > 0.5).astype(np.uint8)
            else:
                base_mask = (sub_mask > 0).astype(np.uint8)
            # Dilate mask slightly to accommodate magnified motion range without clipping
            dilated_mask = cv2.dilate(
                base_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            )
            target_matte = cv2.GaussianBlur(dilated_mask.astype(np.float32), (21, 21), 5.0)
            matte_weight = np.clip(target_matte, 0.0, 1.0)
        else:
            # Feather smoothly near the outer margins of the cropped ROI
            fade = min(12, max(4, margin // 2))
            target_matte = np.zeros((crop_h, crop_w), dtype=np.float32)
            target_matte[fade : crop_h - fade, fade : crop_w - fade] = 1.0
            matte_blur = cv2.GaussianBlur(target_matte, (17, 17), 5.0)
            matte_weight = np.clip(matte_blur, 0.0, 1.0)

        # Blend magnified luminance with original luminance
        blended_lum = matte_weight * mag_lum + (1.0 - matte_weight) * lum_seq

        # Recombine magnified crops into full frames
        mag_full_frames: List[np.ndarray] = []
        for t in range(n_frames):
            yuv = yuv_crops[t].copy()
            yuv[:, :, 0] = np.clip(blended_lum[t], 0, 255)
            bgr_crop = cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
            full_frame = raw_frames[t].copy()
            full_frame[cy0:cy1, cx0:cx1] = bgr_crop
            mag_full_frames.append(full_frame)

        # Render Side-by-Side MP4 with text overlay
        out_w = orig_w * 2
        out_h = orig_h

        font = cv2.FONT_HERSHEY_SIMPLEX
        label_text = f"{freq_center:.1f} Hz | {int(round(alpha))}x magnified"

        composite_frames: List[np.ndarray] = []
        for t in range(n_frames):
            orig_f = raw_frames[t]
            mag_f = mag_full_frames[t]
            sbs = np.hstack([orig_f, mag_f])

            # Vertical divider line
            sbs[:, orig_w - 1 : orig_w + 1] = (50, 50, 50)

            # Left overlay: "Original"
            cv2.rectangle(sbs, (16, 14), (160, 48), (15, 23, 42), -1)
            cv2.putText(
                sbs, "Original", (24, 38), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Right overlay: frequency + "50x magnified"
            cv2.rectangle(sbs, (orig_w + 16, 14), (orig_w + 350, 48), (15, 23, 42), -1)
            cv2.putText(
                sbs,
                label_text,
                (orig_w + 24, 38),
                font,
                0.7,
                (0, 230, 255),
                2,
                cv2.LINE_AA,
            )

            composite_frames.append(sbs)

        # Write H.264 MP4 (yuv420p, +faststart) using PyAV or imageio-ffmpeg
        _write_mp4_video(output_video_path, composite_frames, fps)

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
    )
