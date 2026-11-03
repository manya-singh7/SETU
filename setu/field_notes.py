"""Convert driver transcripts into evidence-locked engineer tags."""
import json
import logging
from typing import Any, Dict

from setu.meta_client import MetaClient, get_meta_client

logger = logging.getLogger(__name__)
ALLOWED_TAGS = {"crack", "noise_under_load", "visible_sag", "rust", "spalling", "water_or_scour", "other"}


def extract_field_note(transcript: str, client: MetaClient = None) -> Dict[str, Any]:
    client = client or get_meta_client()
    if client.mode == "mock":
        data = {"english": None, "tags": [
            {"tag": "crack", "evidence": "crack"},
            {"tag": "rust", "evidence": "rust near the water"},
            {"tag": "spalling", "evidence": "fabricated quote"},
        ]}
        return {"english": data["english"], "tags": [
            t for t in data["tags"] if t["tag"] in ALLOWED_TAGS and t["evidence"] in transcript
        ]}
    prompt = (
        "Return JSON with english (English translation, or null when transcript is English) and tags. "
        "Each tag is {tag,evidence}; tags must be from crack, noise_under_load, visible_sag, rust, "
        "spalling, water_or_scour, other. Evidence must be an exact quote copied from the transcript. "
        "Do not infer safety or structural verdicts. Transcript: " + json.dumps(transcript, ensure_ascii=False)
    )
    response = client.chat_completion(
        [{"role": "user", "content": prompt}], response_format={"type": "json_object"},
        fixture_name="muse_spark_field_note.json", extra_inputs={"transcript": transcript},
    )
    raw = response.get("_mock_data")
    if raw is None:
        raw = response["choices"][0]["message"]["content"]
    data = raw if isinstance(raw, dict) else json.loads(raw)
    accepted = []
    for item in data.get("tags", []):
        tag, evidence = item.get("tag"), item.get("evidence")
        if tag in ALLOWED_TAGS and isinstance(evidence, str) and evidence and evidence in transcript:
            accepted.append({"tag": tag, "evidence": evidence})
    return {"english": data.get("english"), "tags": accepted}
