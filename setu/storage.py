"""
Storage management and path traversal protection for Project SETU.

Handles:
- Scan directory organization: data/scans/<scan_id>/
- Filename sanitization
- Extension validation (.mp4, .mov, .webm)
- File size limit validation (max 200 MB)
- Secure file serving with strict path-traversal prevention
- Scan result JSON persistence
"""

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException, UploadFile, status

# Allowed video extensions
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}

# Maximum video upload size in bytes (200 MB)
MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024

# HTTP status for content too large (413)
HTTP_413 = 413

# Base storage directories
BASE_DIR = Path(os.getcwd())
DATA_DIR = BASE_DIR / "data"
SCANS_DIR = DATA_DIR / "scans"


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal or invalid characters.

    Extracts base filename, removes path components, cleans special characters,
    and preserves a safe lowercase extension.
    """
    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base)

    clean_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem).strip("._")
    if not clean_stem:
        clean_stem = "upload"

    clean_ext = re.sub(r"[^a-zA-Z0-9.]", "", ext).lower()
    return f"{clean_stem}{clean_ext}"


def validate_scan_id(scan_id: str) -> str:
    """Validate scan_id to protect against directory traversal or injection."""
    if not scan_id or not re.match(r"^[a-zA-Z0-9_-]+$", scan_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scan ID format: '{scan_id}'",
        )
    return scan_id


def get_scan_dir(scan_id: str) -> Path:
    """Get validated directory path for a scan under data/scans/<scan_id>/."""
    validate_scan_id(scan_id)
    scan_dir = (SCANS_DIR / scan_id).resolve()

    # Verify scan_dir is strictly contained within SCANS_DIR
    try:
        scan_dir.relative_to(SCANS_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Directory traversal detected in scan ID",
        )

    scan_dir.mkdir(parents=True, exist_ok=True)
    return scan_dir


async def save_uploaded_video(
    upload_file: UploadFile,
    scan_id: str,
) -> Tuple[Path, str]:
    """Validate extension, size, sanitize filename, and store under data/scans/<scan_id>/.

    Raises:
        HTTPException 400: If file extension is not supported.
        HTTPException 413: If file size exceeds MAX_UPLOAD_SIZE_BYTES (200 MB).
    """
    original_filename = upload_file.filename or "video.mp4"
    _, ext = os.path.splitext(original_filename)
    ext_lower = ext.lower()

    if ext_lower not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported video extension '{ext}'. "
                f"Allowed extensions are: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # Pre-check size header if available
    if upload_file.size is not None and upload_file.size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=HTTP_413,
            detail="Uploaded file exceeds the maximum allowed size of 200 MB",
        )

    clean_filename = sanitize_filename(original_filename)
    scan_dir = get_scan_dir(scan_id)
    dest_path = scan_dir / clean_filename

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1 MB chunk

    try:
        with open(dest_path, "wb") as buffer:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    buffer.close()
                    if dest_path.exists():
                        dest_path.unlink()
                    raise HTTPException(
                        status_code=HTTP_413,
                        detail="Uploaded file exceeds the maximum allowed size of 200 MB",
                    )
                buffer.write(chunk)
    finally:
        await upload_file.seek(0)

    return dest_path, clean_filename


def get_scan_file_path(scan_id: str, filename: str) -> Path:
    """Resolve and validate a file within data/scans/<scan_id>/ with path traversal protection.

    Raises:
        HTTPException 400: If path traversal is detected.
        HTTPException 404: If file does not exist.
    """
    validate_scan_id(scan_id)

    # Disallow path navigation tokens or directory delimiters in filename parameter
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal detected in file name",
        )

    scan_dir = (SCANS_DIR / scan_id).resolve()
    target_path = (scan_dir / filename).resolve()

    try:
        target_path.relative_to(scan_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path traversal detected",
        )

    if not target_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' not found for scan '{scan_id}'",
        )

    return target_path


def get_scan_result_path(scan_id: str) -> Path:
    """Return path to data/scans/<scan_id>/result.json."""
    scan_dir = get_scan_dir(scan_id)
    return scan_dir / "result.json"


def save_scan_result(scan_id: str, result_dict: Dict[str, Any]) -> Path:
    """Save scan result JSON to data/scans/<scan_id>/result.json."""
    result_path = get_scan_result_path(scan_id)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    return result_path


def load_scan_result(scan_id: str) -> Optional[Dict[str, Any]]:
    """Load scan result JSON from data/scans/<scan_id>/result.json if present."""
    validate_scan_id(scan_id)
    scan_dir = (SCANS_DIR / scan_id).resolve()
    result_path = scan_dir / "result.json"

    if not result_path.is_file():
        return None

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

