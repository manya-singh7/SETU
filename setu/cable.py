"""Cable scan identification and uncertainty-aware tension-change estimates."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Optional, Tuple

from setu.report import bridge_key_for_scan

CABLE_PROMPT_TERMS = ("cable", "stay", "hanger", "suspender")


def target_prompt(scan: Dict[str, Any]) -> str:
    """Return the scan's target prompt from the shared result shape."""
    return str((scan.get("segmentation") or {}).get("target_prompt")
               or scan.get("target_prompt") or "").strip()


def is_cable_scan(scan_or_prompt: Any, explicit: Optional[bool] = None) -> bool:
    """Identify cable scans by explicit opt-in or the target concept prompt."""
    if isinstance(scan_or_prompt, dict):
        current_flag = (scan_or_prompt.get("cable") or {}).get("is_cable")
        prompt = target_prompt(scan_or_prompt)
        if explicit is None and current_flag is True:
            explicit = True
    else:
        prompt = str(scan_or_prompt or "")
    return explicit is True or any(term in prompt.casefold() for term in CABLE_PROMPT_TERMS)


def _verdict_level(scan: Dict[str, Any]) -> Optional[str]:
    verdict = scan.get("verdict") or {}
    return verdict.get("level") if isinstance(verdict, dict) else str(verdict)


def _parse_time(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    except ValueError:
        return None


def tension_change(
    old_frequency_hz: float,
    new_frequency_hz: float,
    old_ci_hz: Iterable[float],
    new_ci_hz: Iterable[float],
) -> Tuple[float, Tuple[float, float]]:
    """Return percent tension change and propagated 95% uncertainty range.

    Frequency confidence half-widths are combined in quadrature as relative
    errors. Since tension is proportional to frequency squared, the relative
    tension uncertainty is approximately twice that combined relative error.
    """
    old_f, new_f = float(old_frequency_hz), float(new_frequency_hz)
    old_ci, new_ci = [float(x) for x in old_ci_hz], [float(x) for x in new_ci_hz]
    if len(old_ci) != 2 or len(new_ci) != 2:
        raise ValueError("Frequency confidence intervals must have two endpoints.")
    if not all(math.isfinite(v) for v in [old_f, new_f, *old_ci, *new_ci]):
        raise ValueError("Frequencies and confidence intervals must be finite.")
    if old_f <= 0 or new_f <= 0 or old_ci[0] <= 0 or new_ci[0] <= 0:
        raise ValueError("Frequencies and lower confidence limits must be positive.")
    if old_ci[0] > old_ci[1] or new_ci[0] > new_ci[1]:
        raise ValueError("Confidence interval lower endpoints must not exceed upper endpoints.")

    ratio_squared = (new_f / old_f) ** 2
    change_pct = (ratio_squared - 1.0) * 100.0
    old_rel_error = ((old_ci[1] - old_ci[0]) / 2.0) / old_f
    new_rel_error = ((new_ci[1] - new_ci[0]) / 2.0) / new_f
    combined_rel_error = math.hypot(old_rel_error, new_rel_error)
    change_error_pct = ratio_squared * 2.0 * combined_rel_error * 100.0
    return round(change_pct, 4), (
        round(change_pct - change_error_pct, 4),
        round(change_pct + change_error_pct, 4),
    )


def _frequency_and_ci(scan: Dict[str, Any]) -> Tuple[Optional[float], Optional[list]]:
    measurement = scan.get("measurement") or {}
    freq = measurement.get("dominant_frequency_hz")
    ci = measurement.get("frequency_ci_hz")
    try:
        freq = float(freq) if freq is not None else None
        ci = [float(v) for v in ci] if ci is not None else None
    except (TypeError, ValueError):
        return None, None
    if (freq is None or not math.isfinite(freq) or freq <= 0 or ci is None
            or len(ci) != 2 or not all(math.isfinite(v) for v in ci)
            or ci[0] <= 0 or ci[0] > ci[1]):
        return None, None
    return freq, ci


def apply_cable_comparison(
    current: Dict[str, Any],
    earlier_scans: Iterable[Dict[str, Any]],
    explicit_is_cable: Optional[bool] = None,
) -> Dict[str, Any]:
    """Populate schema cable comparison data for a completed scan in place."""
    cable = {
        "is_cable": is_cable_scan(current, explicit=explicit_is_cable),
        "tension_change_pct": None,
        "tension_change_range_pct": None,
        "compared_to_scan_id": None,
        "compared_to_date": None,
    }
    current["cable"] = cable
    if not cable["is_cable"] or current.get("is_demo") is True or _verdict_level(current) != "usable":
        return cable

    current_time = _parse_time(current.get("created_at"))
    current_freq, current_ci = _frequency_and_ci(current)
    if current_time is None or current_freq is None or current_ci is None:
        return cable
    current_prompt = " ".join(target_prompt(current).casefold().split())
    current_bridge = bridge_key_for_scan(current)
    candidates = []
    for prior in earlier_scans:
        if not isinstance(prior, dict) or prior.get("scan_id") == current.get("scan_id"):
            continue
        if prior.get("is_demo") is True or _verdict_level(prior) != "usable":
            continue
        if not is_cable_scan(prior):
            continue
        if bridge_key_for_scan(prior) != current_bridge:
            continue
        if " ".join(target_prompt(prior).casefold().split()) != current_prompt:
            continue
        prior_time = _parse_time(prior.get("created_at"))
        if prior_time is None or prior_time >= current_time:
            continue
        candidates.append((prior_time, prior))

    if not candidates:
        return cable
    _, prior = max(candidates, key=lambda row: row[0])
    prior_freq, prior_ci = _frequency_and_ci(prior)
    if prior_freq is None or prior_ci is None:
        return cable
    try:
        change, uncertainty_range = tension_change(prior_freq, current_freq, prior_ci, current_ci)
    except ValueError:
        return cable
    cable.update({
        "tension_change_pct": change,
        "tension_change_range_pct": list(uncertainty_range),
        "compared_to_scan_id": prior.get("scan_id"),
        "compared_to_date": prior.get("created_at"),
    })
    return cable
