"""
Subpixel structural motion tracking and camera jitter cancellation module for Project SETU.

Step 3 in SETU Pipeline:
1. Stabilises video frames by estimating per-frame affine transformations from
   features in the stationary background mask via RANSAC and warping each frame
   to reference frame 0.
2. Measures structural target motion using a FIXED mask from frame 0 (since the
   target vibrates subpixel).
3. Primary method: Phase correlation on the masked target region along the
   dominant motion axis.
4. Secondary method: Subpixel Lucas-Kanade optical flow on good features inside
   the target mask.
5. Computes quality diagnostics: background residual RMS, estimated displacement
   SNR, pixel saturation/clipping check, and warnings for excessive camera tremor.
6. Three-level verdict: "usable", "weak", or "failed", with diagnostic reasons.

Note on Physical Units:
   Displacement is reported strictly in pixel units (units="px", scale=None).
   Converting pixel displacement to physical millimetres (mm) requires a known
   physical reference length in the camera plane at the bridge structure (e.g.
   pillar diameter, bolt spacing, or surveyor scale bar) to calibrate pixels per
   millimetre.

Saturation / Clipping Threshold Justification:
   Subpixel motion estimation via phase correlation and Lucas-Kanade differential optical
   flow relies on continuous spatial intensity gradients (grad I * dx ~ dI). When sensor
   saturation occurs (pixel values at 0 or 255), grad I -> 0, causing flat-top clipping
   that suppresses high spatial frequency features and introduces nonlinear harmonic distortion.
   - SATURATION_PIXEL_FRACTION = 0.15: If more than 15% of pixels in the target ROI are saturated
     in a frame, the gradient field is critically impaired.
   - CLIPPING_FRAME_FRACTION = 0.10: If more than 10% of frames in the recording suffer from target
     clipping, subpixel tracking accuracy degrades throughout the displacement time series,
     warranting a 'weak' verdict reason.

Classification Safety Principle:
   Weak input must NEVER be described as an anomaly. A "weak" verdict reflects sensor dynamic range,
   measurement SNR, or environmental reference limitations requiring engineer triage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import cv2
import numpy as np
import numpy.typing as npt
import av

from setu.types import MotionResult, SegmentResult

SATURATION_PIXEL_FRACTION: float = 0.15
"""Fraction of pixels inside the target mask that must be saturated (intensity <= 5 or >= 250)
for a frame to be marked as clipped.
"""

CLIPPING_FRAME_FRACTION: float = 0.10
"""Fraction of total video frames that must exhibit target pixel clipping to trigger a 'weak'
verdict reason for sensor saturation.
"""

TRACKING_NOISE_FLOOR_RMS_PX: float = 0.008
"""Empirical subpixel tracking noise floor RMS in pixels, measured on a synthetic video
of a completely still target under H.264 compression (CRF 23).
"""

NOISE_FLOOR_FACTOR: float = 3.0
"""Safety factor above the tracking noise floor required to confirm measurable physical motion
(justified by the Rose detection criterion and 3-sigma IUPAC limit of detection).
"""


def _extract_bounding_box(mask: npt.NDArray[np.uint8]) -> Tuple[int, int, int, int]:
    """Find the bounding box (x, y, w, h) of a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(c)
    # Fallback to whole mask dimensions
    h, w = mask.shape[:2]
    return (0, 0, w, h)


def _select_phase_patches(
    gray: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_]
) -> List[Tuple[int, int, int, int]]:
    """Select three disjoint, textured DFT-friendly windows when the ROI allows."""
    height, width = gray.shape[:2]

    def optimal_not_larger_than(limit: int) -> int:
        size = min(128, limit)
        while size > 16 and cv2.getOptimalDFTSize(size) > limit:
            size -= 1
        return cv2.getOptimalDFTSize(size) if size > 16 else limit

    # Small segmentation boxes can include stationary pixels beside a moving
    # structure. Keep those on the full-ROI estimator; use compact patches only
    # when the ROI is large enough to provide separated interior windows.
    desired_size = 64 if min(width, height) >= 192 else 128
    patch_size = min(desired_size, optimal_not_larger_than(min(width, height)))
    patch_w = patch_h = patch_size
    if patch_w < 16 or patch_h < 16:
        return [(0, 0, width, height)]

    image = gray.astype(np.float32)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    candidates: List[Tuple[float, int, int]] = []
    # Tiling makes candidates non-overlapping by construction. Require each
    # candidate window to lie inside the supplied target mask.
    for y in range(0, height - patch_h + 1, patch_h):
        for x in range(0, width - patch_w + 1, patch_w):
            patch_mask = mask[y : y + patch_h, x : x + patch_w]
            if not np.all(patch_mask):
                continue
            candidates.append((float(np.sum(energy[y : y + patch_h, x : x + patch_w])), x, y))

    candidates.sort(reverse=True)
    if len(candidates) < 3:
        return [(0, 0, width, height)]
    return [(x, y, patch_w, patch_h) for _, x, y in candidates[:3]]


def track_motion(
    video_path: str,
    segment_result: Optional[SegmentResult] = None,
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    background_bbox: Optional[Tuple[int, int, int, int]] = None,
    method: Literal["phase_correlation", "lucas_kanade"] = "phase_correlation",
    axis: Literal["auto", "x", "y"] = "auto",
    max_corners_bg: int = 150,
    max_corners_target: int = 100,
    noise_floor_rms: float = TRACKING_NOISE_FLOOR_RMS_PX,
    noise_factor: float = NOISE_FLOOR_FACTOR,
) -> MotionResult:
    """Track subpixel target motion cancelled against background micro-jitter.

    Physical Units Note:
        Displacement is in pixels (units='px', scale=None). Converting to millimetres (mm)
        requires a known physical reference length in the camera plane.

    Args:
        video_path: Path to input MP4 video file.
        segment_result: Optional SegmentResult containing target and background masks.
        target_bbox: Optional (x, y, w, h) bounding box for target if segment_result is omitted.
        background_bbox: Optional (x, y, w, h) bounding box for background if segment_result is omitted.
        method: Motion extraction technique ('phase_correlation' or 'lucas_kanade').
        axis: Motion projection axis ('auto', 'x', 'y'). 'auto' selects higher-variance axis.
        max_corners_bg: Maximum corner features to track in the background reference.
        max_corners_target: Maximum corner features for Lucas-Kanade target tracking.

    Returns:
        MotionResult containing 1D displacement time series, frame rate, and quality diagnostics.
    """
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 60.0)
        width, height = int(stream.width), int(stream.height)
        decoded = container.decode(stream)
        frame0_av = next(decoded, None)
        if frame0_av is None:
            raise ValueError(f"Failed to read reference frame 0 from '{video_path}'")
        gray0 = frame0_av.to_ndarray(format="gray")
    except Exception as exc:
        raise ValueError(f"Could not open video file: '{video_path}'") from exc
    warnings: List[str] = []

    # 1. Establish frame 0 target and background masks
    has_background = True
    bg_mask0 = None
    b_box = background_bbox

    if segment_result is not None:
        if len(segment_result.target_masks) > 0:
            target_mask0 = segment_result.target_masks[0].astype(np.uint8) * 255
            t_box = target_bbox or _extract_bounding_box(target_mask0)
        else:
            t_box = target_bbox or (int(width * 0.4), int(height * 0.2), int(width * 0.2), int(height * 0.6))
            target_mask0 = np.zeros((height, width), dtype=np.uint8)
            tx, ty, tw, th = t_box
            target_mask0[ty : ty + th, tx : tx + tw] = 255

        if len(segment_result.background_masks) > 0 and np.any(segment_result.background_masks[0]):
            bg_mask0 = segment_result.background_masks[0].astype(np.uint8) * 255
            b_box = b_box or _extract_bounding_box(bg_mask0)
        elif b_box is not None and b_box[2] > 0 and b_box[3] > 0:
            bx, by, bw, bh = b_box
            bg_mask0 = np.zeros((height, width), dtype=np.uint8)
            bg_mask0[by : by + bh, bx : bx + bw] = 255
        else:
            has_background = False
    else:
        t_box = target_bbox or (int(width * 0.4), int(height * 0.2), int(width * 0.2), int(height * 0.6))
        target_mask0 = np.zeros((height, width), dtype=np.uint8)
        tx, ty, tw, th = t_box
        target_mask0[ty : ty + th, tx : tx + tw] = 255

        if b_box is not None and b_box[2] > 0 and b_box[3] > 0:
            bx, by, bw, bh = b_box
            bg_mask0 = np.zeros((height, width), dtype=np.uint8)
            bg_mask0[by : by + bh, bx : bx + bw] = 255
        else:
            has_background = False

    tx, ty, tw, th = t_box

    # Target mask clipping check setup on Frame 0
    target_roi_mask = (target_mask0[ty : ty + th, tx : tx + tw] > 0)
    roi0 = gray0[ty : ty + th, tx : tx + tw]
    pix0 = roi0[target_roi_mask] if np.any(target_roi_mask) else roi0.ravel()
    sat_count0 = int(np.sum((pix0 <= 5) | (pix0 >= 250)))
    clipped_frames_count = 1 if (sat_count0 / max(len(pix0), 1) >= SATURATION_PIXEL_FRACTION) else 0
    total_frames_count = 1

    # 2. Setup background feature tracking (or skip if no background available)
    if not has_background or bg_mask0 is None or not np.any(bg_mask0):
        has_background = False
        warnings.append("No stationary background found; skipping camera micro-jitter stabilisation.")
        p_bg0 = None
    else:
        bx, by, bw, bh = b_box
        bg_roi0 = gray0[by:by + bh, bx:bx + bw]
        bg_mask_roi = bg_mask0[by:by + bh, bx:bx + bw]
        p_bg0 = cv2.goodFeaturesToTrack(
            bg_roi0,
            mask=bg_mask_roi,
            maxCorners=max_corners_bg,
            qualityLevel=0.01,
            minDistance=6,
        )

        if p_bg0 is None or len(p_bg0) < 6:
            warnings.append(
                f"Sparse background features ({len(p_bg0) if p_bg0 is not None else 0} detected). "
                "Micro-jitter cancellation stability may be reduced."
            )
            # Fallback to general grid points inside background box
            ys, xs = np.mgrid[b_box[1] + 10 : b_box[1] + b_box[3] - 10 : 15, b_box[0] + 10 : b_box[0] + b_box[2] - 10 : 15]
            grid_pts = np.vstack((xs.flatten() - b_box[0], ys.flatten() - b_box[1])).T.astype(np.float32)
            p_bg0 = grid_pts[:, np.newaxis, :]
        else:
            p_bg0 = cv2.cornerSubPix(
                bg_roi0,
                p_bg0,
                (5, 5),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

    # 3. Setup method-specific targets
    if method == "phase_correlation":
        target_roi_gray = gray0[ty : ty + th, tx : tx + tw]
        phase_patches = _select_phase_patches(target_roi_gray, target_roi_mask)
        phase_references = []
        phase_windows = []
        for px, py, pw, ph in phase_patches:
            phase_references.append(target_roi_gray[py : py + ph, px : px + pw].astype(np.float32))
            phase_windows.append(cv2.createHanningWindow((pw, ph), cv2.CV_32F))
    else:
        # Lucas-Kanade target points
        p_tgt0 = cv2.goodFeaturesToTrack(
            gray0,
            mask=target_mask0,
            maxCorners=max_corners_target,
            qualityLevel=0.01,
            minDistance=5,
        )
        if p_tgt0 is not None and len(p_tgt0) > 0:
            p_tgt0 = cv2.cornerSubPix(
                gray0,
                p_tgt0,
                (5, 5),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
        else:
            warnings.append("No distinct corners found inside target mask for Lucas-Kanade; using center grid.")
            ys, xs = np.mgrid[ty + 10 : ty + th - 10 : 10, tx + 10 : tx + tw - 10 : 10]
            grid_pts = np.vstack((xs.flatten(), ys.flatten())).T.astype(np.float32)
            p_tgt0 = grid_pts[:, np.newaxis, :]

    # 4. Process all frames
    dx_raw_list: List[float] = [0.0]
    dy_raw_list: List[float] = [0.0]
    bg_residuals: List[float] = [0.0]
    camera_shifts_x: List[float] = [0.0]
    camera_shifts_y: List[float] = [0.0]
    background_displacement: List[float] = [0.0]
    M_prev = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    bx, by, bw, bh = b_box if b_box is not None else (0, 0, 0, 0)
    try:
      for frame_av in decoded:
        gray_curr = frame_av.to_ndarray(format="gray")
        if gray_curr is None:
            break

        total_frames_count += 1

        # Target mask saturation/clipping check on current frame
        roi_curr = gray_curr[ty : ty + th, tx : tx + tw]
        pix_curr = roi_curr[target_roi_mask] if np.any(target_roi_mask) else roi_curr.ravel()
        sat_count = int(np.sum((pix_curr <= 5) | (pix_curr >= 250)))
        if (sat_count / max(len(pix_curr), 1)) >= SATURATION_PIXEL_FRACTION:
            clipped_frames_count += 1

        if has_background and p_bg0 is not None:
            # A. Track background features from frame 0 to current frame
            p_bg_curr, st_bg, _ = cv2.calcOpticalFlowPyrLK(
                gray0[by:by + bh, bx:bx + bw],
                gray_curr[by:by + bh, bx:bx + bw],
                p_bg0,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            st_bg_flat = (st_bg.ravel() == 1) if st_bg is not None else np.zeros(len(p_bg0), dtype=bool)
            good_bg0 = p_bg0[st_bg_flat]
            good_bg_curr = p_bg_curr[st_bg_flat]
            good_bg0 = good_bg0 + np.array([bx, by], dtype=np.float32)
            good_bg_curr = good_bg_curr + np.array([bx, by], dtype=np.float32)

            # B. Estimate affine warp matrix to align current frame to frame 0
            if len(good_bg_curr) >= 4:
                M, inliers = cv2.estimateAffinePartial2D(
                    good_bg_curr, good_bg0, method=cv2.RANSAC, ransacReprojThreshold=2.0
                )
                if M is None:
                    M = M_prev.copy()
                else:
                    trans_jump = float(np.linalg.norm(M[:, 2] - M_prev[:, 2]))
                    inlier_ratio = (
                        float(np.sum(inliers)) / float(len(good_bg_curr)) if inliers is not None else 0.0
                    )
                    if trans_jump > 3.0 or inlier_ratio < 0.25:
                        warnings.append(
                            f"Frame stabilization jump ({trans_jump:.1f}px) or low inlier ratio ({inlier_ratio:.2f}); "
                            "applying temporal continuity."
                        )
                        M = M_prev.copy()
                    else:
                        M_prev = M.copy()
            else:
                M = M_prev.copy()
                warnings.append("Insufficient tracked background points in frame; maintaining prior alignment.")

            # Record camera motion translation from M
            camera_shifts_x.append(float(M[0, 2]))
            camera_shifts_y.append(float(M[1, 2]))

            # C. Warp current frame to frame 0 reference coordinate system
            roi_M = M.copy()
            # Output coordinates are ROI-local, while source coordinates remain
            # full-frame coordinates: translate the destination by the ROI origin.
            roi_M[:, 2] -= np.array([tx, ty], dtype=roi_M.dtype)
            warped_curr = cv2.warpAffine(
                gray_curr,
                roi_M,
                (tw, th),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            # D. Calculate background residual error after stabilization
            if len(good_bg_curr) > 0 and M is not None:
                pts_reshaped = good_bg_curr.reshape(-1, 2)
                # Apply M to current background points
                warped_bg_pts = (M[:, :2] @ pts_reshaped.T).T + M[:, 2]
                res_diff = warped_bg_pts - good_bg0.reshape(-1, 2)
                res_rms = float(np.sqrt(np.mean(np.sum(res_diff**2, axis=1))))
                bg_residuals.append(res_rms)
                background_displacement.append(
                    float(np.median(np.linalg.norm(res_diff, axis=1)))
                )
            else:
                bg_residuals.append(0.0)
                background_displacement.append(0.0)
        else:
            # Skip camera stabilization: use unwarped frame and record zero camera shifts
            warped_curr = gray_curr[ty : ty + th, tx : tx + tw]
            camera_shifts_x.append(0.0)
            camera_shifts_y.append(0.0)
            bg_residuals.append(0.0)
            background_displacement.append(0.0)

        # E. Measure target motion using FIXED frame 0 mask
        if method == "phase_correlation":
            shifts = []
            for (px, py, pw, ph), reference, window in zip(phase_patches, phase_references, phase_windows):
                current_patch = warped_curr[py : py + ph, px : px + pw].astype(np.float32)
                shift, _ = cv2.phaseCorrelate(reference, current_patch, window)
                shifts.append(shift)
            dx_step, dy_step = np.median(np.asarray(shifts, dtype=np.float32), axis=0)
            dx_raw_list.append(float(dx_step))
            dy_raw_list.append(float(dy_step))
        else:
            # Lucas-Kanade on stabilized target features
            p_tgt_curr, st_tgt, _ = cv2.calcOpticalFlowPyrLK(
                gray0,
                warped_curr if not has_background else gray_curr,
                p_tgt0,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            st_tgt_flat = (st_tgt.ravel() == 1) if st_tgt is not None else np.zeros(len(p_tgt0), dtype=bool)
            valid0 = p_tgt0[st_tgt_flat].reshape(-1, 2)
            valid_curr = p_tgt_curr[st_tgt_flat].reshape(-1, 2)

            if len(valid_curr) > 0:
                diff = valid_curr - valid0
                dx_step = float(np.median(diff[:, 0]))
                dy_step = float(np.median(diff[:, 1]))
            else:
                dx_step = 0.0
                dy_step = 0.0

            dx_raw_list.append(dx_step)
            dy_raw_list.append(dy_step)

    finally:
        container.close()

    dx_arr = np.array(dx_raw_list, dtype=np.float64)
    dy_arr = np.array(dy_raw_list, dtype=np.float64)

    # 5. Determine dominant motion axis
    var_x = float(np.var(dx_arr))
    var_y = float(np.var(dy_arr))

    if axis == "x":
        displacement = dx_arr
    elif axis == "y":
        displacement = dy_arr
    else:
        displacement = dx_arr if var_x >= var_y else dy_arr

    # 6. Quality diagnostics
    bg_residual_px = float(np.mean(bg_residuals))
    is_valid_disp = len(displacement) > 0 and bool(np.all(np.isfinite(displacement)))
    if is_valid_disp:
        disp_std = float(np.std(displacement))
        disp_rms = float(np.sqrt(np.mean(displacement**2)))
    else:
        disp_std = 0.0
        disp_rms = 0.0

    # Signal to noise ratio calculation
    noise_est = max(bg_residual_px, 1e-4)
    snr_val = float(np.clip((disp_std / noise_est) ** 2, 0.01, 1e6)) if is_valid_disp else 0.0

    # Check if camera motion is substantially larger than target structural motion
    cam_rms = float(np.sqrt(np.mean(np.array(camera_shifts_x) ** 2 + np.array(camera_shifts_y) ** 2)))
    if is_valid_disp and cam_rms > 5.0 * max(disp_std, 0.05):
        warnings.append(
            f"Camera micro-jitter motion (RMS: {cam_rms:.2f} px) is significantly larger than "
            f"target structural displacement (RMS: {disp_std:.3f} px). Subpixel accuracy may be degraded."
        )

    if bg_residual_px > 0.08:
        warnings.append(
            f"Elevated background stabilization residual ({bg_residual_px:.3f} px). "
            "High-amplitude ground or mount vibration detected."
        )

    # Clipping detection evaluation
    clipped_frame_ratio = float(clipped_frames_count / max(total_frames_count, 1))
    clipping_detected = bool(clipped_frame_ratio >= CLIPPING_FRAME_FRACTION)
    if clipping_detected:
        warnings.append(
            f"Target mask pixel saturation/clipping detected in {clipped_frame_ratio * 100:.1f}% of frames "
            f"(threshold: {CLIPPING_FRAME_FRACTION * 100:.0f}%). Saturated pixel dynamic range (0 or 255) "
            "reduces sub-pixel gradient accuracy."
        )

    # Three-level verdict determination
    reasons: List[str] = []
    if not has_background:
        reasons.append("unverified: no stationary background in frame")
    if clipping_detected:
        reasons.append("target mask pixel saturation/clipping detected (saturation reduces sub-pixel accuracy)")
    if is_valid_disp and cam_rms > 5.0 * max(disp_std, 0.05):
        reasons.append(
            f"camera micro-jitter motion ({cam_rms:.2f} px) significantly exceeds target motion ({disp_std:.3f} px)"
        )
    if bg_residual_px > 0.08:
        reasons.append(f"elevated background stabilization residual ({bg_residual_px:.3f} px)")

    min_motion_thresh = float(noise_floor_rms * noise_factor)

    if not is_valid_disp:
        verdict: Literal["usable", "weak", "failed"] = "failed"
        reasons.insert(0, "Motion tracking produced invalid values")
    elif disp_rms <= min_motion_thresh:
        verdict = "failed"
        reasons.insert(0, "No measurable motion above tracking noise")
    elif len(reasons) > 0:
        verdict = "weak"
    else:
        verdict = "usable"

    quality: Dict[str, Any] = {
        "verdict": verdict,
        "reasons": reasons,
        "units": "px",
        "scale": None,
        "has_background": has_background,
        "clipping_detected": clipping_detected,
        "clipped_frame_ratio": round(clipped_frame_ratio, 4),
        "background_residual_px": round(bg_residual_px, 4),
        "snr": round(snr_val, 2),
        "camera_motion_rms_px": round(cam_rms, 4),
        "dominant_axis": "x" if (axis == "x" or (axis == "auto" and var_x >= var_y)) else "y",
        "method": method,
        "warnings": warnings,
        "confidence": verdict,
        "displacement_rms_px": round(disp_rms, 6),
        "noise_floor_rms_px": round(noise_floor_rms, 6),
        "min_motion_threshold_px": round(min_motion_thresh, 6),
    }

    motion_result = MotionResult(
        displacement=displacement,
        fps=fps,
        quality=quality,
        units="px",
        scale=None,
        verdict=verdict,
        reasons=reasons,
    )
    motion_result.background_displacement = np.asarray(background_displacement, dtype=np.float64)

    return motion_result


class SubpixelMotionTracker:
    """Class wrapper for subpixel displacement tracking and camera stabilization."""

    def __init__(
        self,
        fps: float = 60.0,
        method: Literal["phase_correlation", "lucas_kanade"] = "phase_correlation",
        noise_floor_rms: float = TRACKING_NOISE_FLOOR_RMS_PX,
        noise_factor: float = NOISE_FLOOR_FACTOR,
    ) -> None:
        self.fps = fps
        self.method = method
        self.noise_floor_rms = noise_floor_rms
        self.noise_factor = noise_factor

    def track(
        self,
        video_path: str,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        background_bbox: Optional[Tuple[int, int, int, int]] = None,
        noise_floor_rms: Optional[float] = None,
        noise_factor: Optional[float] = None,
    ) -> MotionResult:
        """Execute subpixel motion tracking on a video file."""
        return track_motion(
            video_path=video_path,
            target_bbox=target_bbox,
            background_bbox=background_bbox,
            method=self.method,
            noise_floor_rms=self.noise_floor_rms if noise_floor_rms is None else noise_floor_rms,
            noise_factor=self.noise_factor if noise_factor is None else noise_factor,
        )

    def track_displacement_series(
        self,
        frames: npt.NDArray[np.uint8],
        target_bbox: Tuple[int, int, int, int],
        background_bbox: Tuple[int, int, int, int],
    ) -> MotionResult:
        """In-memory array fallback helper."""
        # For in-memory arrays, write to a temp video and process
        import tempfile
        import os

        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        try:
            h, w = frames.shape[1:3]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_path, fourcc, self.fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            return self.track(temp_path, target_bbox=target_bbox, background_bbox=background_bbox)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
