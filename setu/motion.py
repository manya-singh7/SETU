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
   SNR, and warnings for excessive camera tremor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import cv2
import numpy as np
import numpy.typing as npt

from setu.types import MotionResult, SegmentResult


def _extract_bounding_box(mask: npt.NDArray[np.uint8]) -> Tuple[int, int, int, int]:
    """Find the bounding box (x, y, w, h) of a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(c)
    # Fallback to whole mask dimensions
    h, w = mask.shape[:2]
    return (0, 0, w, h)


def track_motion(
    video_path: str,
    segment_result: Optional[SegmentResult] = None,
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    background_bbox: Optional[Tuple[int, int, int, int]] = None,
    method: Literal["phase_correlation", "lucas_kanade"] = "phase_correlation",
    axis: Literal["auto", "x", "y"] = "auto",
    max_corners_bg: int = 150,
    max_corners_target: int = 100,
) -> MotionResult:
    """Track subpixel target motion cancelled against background micro-jitter.

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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0 or np.isnan(fps):
        fps = 60.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame0 = cap.read()
    if not ret or frame0 is None:
        cap.release()
        raise ValueError(f"Failed to read reference frame 0 from '{video_path}'")

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # 1. Establish frame 0 masks
    if segment_result is not None and len(segment_result.target_masks) > 0:
        target_mask0 = segment_result.target_masks[0].astype(np.uint8) * 255
        bg_mask0 = segment_result.background_masks[0].astype(np.uint8) * 255
        t_box = target_bbox or _extract_bounding_box(target_mask0)
        b_box = background_bbox or _extract_bounding_box(bg_mask0)
    else:
        t_box = target_bbox or (int(width * 0.4), int(height * 0.2), int(width * 0.2), int(height * 0.6))
        b_box = background_bbox or (int(width * 0.05), int(height * 0.7), int(width * 0.25), int(height * 0.2))

        target_mask0 = np.zeros((height, width), dtype=np.uint8)
        tx, ty, tw, th = t_box
        target_mask0[ty : ty + th, tx : tx + tw] = 255

        bg_mask0 = np.zeros((height, width), dtype=np.uint8)
        bx, by, bw, bh = b_box
        bg_mask0[by : by + bh, bx : bx + bw] = 255

    tx, ty, tw, th = t_box
    warnings: List[str] = []

    # 2. Detect good features to track in stationary background mask on frame 0
    p_bg0 = cv2.goodFeaturesToTrack(
        gray0,
        mask=bg_mask0,
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
        grid_pts = np.vstack((xs.flatten(), ys.flatten())).T.astype(np.float32)
        p_bg0 = grid_pts[:, np.newaxis, :]
    else:
        p_bg0 = cv2.cornerSubPix(
            gray0,
            p_bg0,
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    # 3. Setup method-specific targets
    if method == "phase_correlation":
        target_roi0 = gray0[ty : ty + th, tx : tx + tw].astype(np.float64)
        hanning_win = cv2.createHanningWindow((tw, th), cv2.CV_64F)
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
    M_prev = np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    while True:
        ret, frame_curr = cap.read()
        if not ret or frame_curr is None:
            break

        gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY)

        # A. Track background features from frame 0 to current frame
        p_bg_curr, st_bg, _ = cv2.calcOpticalFlowPyrLK(
            gray0,
            gray_curr,
            p_bg0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        st_bg_flat = (st_bg.ravel() == 1) if st_bg is not None else np.zeros(len(p_bg0), dtype=bool)
        good_bg0 = p_bg0[st_bg_flat]
        good_bg_curr = p_bg_curr[st_bg_flat]

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
        warped_curr = cv2.warpAffine(
            gray_curr,
            M,
            (width, height),
            flags=cv2.INTER_LANCZOS4,
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
        else:
            bg_residuals.append(0.0)

        # E. Measure target motion using FIXED frame 0 mask
        if method == "phase_correlation":
            target_roi_curr = warped_curr[ty : ty + th, tx : tx + tw].astype(np.float64)
            (dx_step, dy_step), _ = cv2.phaseCorrelate(target_roi0, target_roi_curr, hanning_win)
            dx_raw_list.append(float(dx_step))
            dy_raw_list.append(float(dy_step))
        else:
            # Lucas-Kanade on stabilized target features
            p_tgt_curr, st_tgt, _ = cv2.calcOpticalFlowPyrLK(
                gray0,
                warped_curr,
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

    cap.release()

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
    disp_std = float(np.std(displacement))

    # Signal to noise ratio calculation
    noise_est = max(bg_residual_px, 1e-4)
    snr_val = float(np.clip((disp_std / noise_est) ** 2, 0.01, 1e6))

    # Check if camera motion is substantially larger than target structural motion
    cam_rms = float(np.sqrt(np.mean(np.array(camera_shifts_x) ** 2 + np.array(camera_shifts_y) ** 2)))
    if cam_rms > 5.0 * max(disp_std, 0.05):
        warnings.append(
            f"Camera micro-jitter motion (RMS: {cam_rms:.2f} px) is significantly larger than "
            f"target structural displacement (RMS: {disp_std:.3f} px). Subpixel accuracy may be degraded."
        )

    if bg_residual_px > 0.08:
        warnings.append(
            f"Elevated background stabilization residual ({bg_residual_px:.3f} px). "
            "High-amplitude ground or mount vibration detected."
        )

    quality: Dict[str, Any] = {
        "background_residual_px": round(bg_residual_px, 4),
        "snr": round(snr_val, 2),
        "camera_motion_rms_px": round(cam_rms, 4),
        "dominant_axis": "x" if (axis == "x" or (axis == "auto" and var_x >= var_y)) else "y",
        "method": method,
        "warnings": warnings,
    }

    return MotionResult(
        displacement=displacement,
        fps=fps,
        quality=quality,
    )


class SubpixelMotionTracker:
    """Class wrapper for subpixel displacement tracking and camera stabilization."""

    def __init__(
        self,
        fps: float = 60.0,
        method: Literal["phase_correlation", "lucas_kanade"] = "phase_correlation",
    ) -> None:
        self.fps = fps
        self.method = method

    def track(
        self,
        video_path: str,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
        background_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> MotionResult:
        """Execute subpixel motion tracking on a video file."""
        return track_motion(
            video_path=video_path,
            target_bbox=target_bbox,
            background_bbox=background_bbox,
            method=self.method,
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
