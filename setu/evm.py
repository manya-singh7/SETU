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
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import numpy.typing as npt
from scipy.signal import butter, filtfilt


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


def _compute_riesz_filters(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute 2D Riesz transform frequency response filters."""
    u = np.fft.fftfreq(width)
    v = np.fft.fftfreq(height)
    u_grid, v_grid = np.meshgrid(u, v)

    denom = np.sqrt(u_grid**2 + v_grid**2)
    denom[0, 0] = 1.0  # Avoid zero division at DC

    rx_filter = -1j * u_grid / denom
    ry_filter = -1j * v_grid / denom

    rx_filter[0, 0] = 0.0
    ry_filter[0, 0] = 0.0
    return rx_filter, ry_filter


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
    order: int = 2,
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

    magnified_pyramids: List[List[Any]] = [
        [None] * (levels + 1) for _ in range(num_frames)
    ]
    # Keep lowpass residual unmagnified
    for t in range(num_frames):
        magnified_pyramids[t][levels] = frame_pyramids[t][levels]

    # 2. Process each spatial bandpass level with Riesz transform
    for lvl in range(levels):
        subbands = np.array([frame_pyramids[t][lvl] for t in range(num_frames)], dtype=np.float64)
        lvl_h, lvl_w = subbands.shape[1], subbands.shape[2]

        rx_filt, ry_filt = _compute_riesz_filters(lvl_w, lvl_h)

        # 2D Riesz transform in frequency domain across frames
        fft_subbands = np.fft.fft2(subbands, axes=(-2, -1))
        rx = np.real(np.fft.ifft2(fft_subbands * rx_filt, axes=(-2, -1)))
        ry = np.real(np.fft.ifft2(fft_subbands * ry_filt, axes=(-2, -1)))

        # Reference subband (frame 0)
        l0 = subbands[0]
        rx0 = rx[0]
        ry0 = ry[0]
        a0_sq = l0**2 + rx0**2 + ry0**2 + 1e-2

        # Instantaneous spatial phase differences (Wadhwa et al. 2014 eq. 4 & 5)
        dphi_x = (l0 * rx - rx0 * subbands) / a0_sq
        dphi_y = (l0 * ry - ry0 * subbands) / a0_sq

        # Temporal bandpass filter around target resonant frequency
        filt_dphi_x = _temporal_bandpass(dphi_x, fps, target_freq_hz, bandwidth_hz)
        filt_dphi_y = _temporal_bandpass(dphi_y, fps, target_freq_hz, bandwidth_hz)

        # Amplify phase variations
        amp_phase_x = alpha * filt_dphi_x
        amp_phase_y = alpha * filt_dphi_y

        amp_magnitude = np.sqrt(amp_phase_x**2 + amp_phase_y**2)
        safe_magnitude = np.maximum(amp_magnitude, 1e-8)
        norm_x = amp_phase_x / safe_magnitude
        norm_y = amp_phase_y / safe_magnitude

        # Reconstruct phase-shifted subband (Wadhwa et al. 2014 eq. 9)
        mag_subbands = np.cos(amp_magnitude) * subbands - np.sin(amp_magnitude) * (
            norm_x * rx + norm_y * ry
        )

        for t in range(num_frames):
            magnified_pyramids[t][lvl] = mag_subbands[t]

    # 3. Collapse pyramids to synthesize magnified frames
    magnified_sequence = np.array(
        [_collapse_laplacian_pyramid(magnified_pyramids[t]) for t in range(num_frames)],
        dtype=np.float64,
    )
    return magnified_sequence


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

        max_frames = int(round(fps * max_duration_sec)) if max_duration_sec else total_frames
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
        alpha = float(amplification_factor if amplification_factor is not None else self.config.amplification_factor)

        # Determine target bounding box for localized CPU processing
        tx, ty, tw, th = (0, 0, orig_w, orig_h)
        if target_bbox is not None:
            tx, ty, tw, th = target_bbox
        elif target_mask is not None:
            contours, _ = cv2.findContours(
                (target_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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
        yuv_crops = np.array([cv2.cvtColor(c, cv2.COLOR_BGR2YUV) for c in crops], dtype=np.float64)
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
                [cv2.resize(lum_seq[t], (proc_w, proc_h), interpolation=cv2.INTER_AREA) for t in range(n_frames)]
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
                [cv2.resize(mag_lum_proc[t], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR) for t in range(n_frames)]
            )
        else:
            mag_lum = mag_lum_proc

        # Spatial blending mask: magnify strictly inside target, smooth transition at borders
        rel_tx = max(0, tx - cx0)
        rel_ty = max(0, ty - cy0)
        rel_tw = min(tw, crop_w - rel_tx)
        rel_th = min(th, crop_h - rel_ty)

        target_matte = np.zeros((crop_h, crop_w), dtype=np.float32)
        if target_mask is not None:
            sub_mask = target_mask[cy0:cy1, cx0:cx1]
            if np.issubdtype(sub_mask.dtype, np.floating):
                target_matte = np.clip(sub_mask, 0.0, 1.0).astype(np.float32)
            else:
                target_matte = (sub_mask > 0).astype(np.float32)
        elif target_bbox is not None:
            target_matte[rel_ty : rel_ty + rel_th, rel_tx : rel_tx + rel_tw] = 1.0
        else:
            target_matte[:, :] = 1.0

        # Smooth edges with Gaussian blur for seamless blending transition
        k_w = min(15, max(3, (crop_w // 4) * 2 + 1))
        k_h = min(15, max(3, (crop_h // 4) * 2 + 1))
        matte_blur = cv2.GaussianBlur(target_matte, (k_w, k_h), 3.0)
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

        # Codec selection: try mp4v first (robust on Windows/Linux without external DLLs), fallback to avc1/H264
        writer: Optional[cv2.VideoWriter] = None
        for codec in ["mp4v", "avc1", "H264"]:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), (out_w, out_h))
                if test_writer.isOpened():
                    writer = test_writer
                    break
            except Exception:
                continue

        if writer is None or not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, float(fps), (out_w, out_h))

        if not writer.isOpened():
            raise RuntimeError(f"Could not initialize VideoWriter for '{output_video_path}'")

        font = cv2.FONT_HERSHEY_SIMPLEX
        label_text = f"{freq_center:.1f} Hz | {int(round(alpha))}x magnified"

        for t in range(n_frames):
            orig_f = raw_frames[t]
            mag_f = mag_full_frames[t]
            sbs = np.hstack([orig_f, mag_f])

            # Vertical divider line
            sbs[:, orig_w - 1 : orig_w + 1] = (50, 50, 50)

            # Left overlay: "Original"
            cv2.rectangle(sbs, (16, 14), (160, 48), (15, 23, 42), -1)
            cv2.putText(sbs, "Original", (24, 38), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

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

            writer.write(sbs)

        writer.release()

        elapsed = time.time() - start_time
        fps_proc = n_frames / elapsed if elapsed > 0 else 0
        print(f"[EVM] Phase-based Riesz magnification completed in {elapsed:.2f}s ({fps_proc:.1f} fps) for {n_frames} frames.")

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
