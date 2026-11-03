import json
from pathlib import Path

import pytest

from scripts import publish_results


def _scan_folder(root: Path, *, status="complete", is_demo=False) -> Path:
    folder = root / "scan-source"
    folder.mkdir(parents=True)
    result = {
        "scan_id": "scan_test",
        "status": status,
        "is_demo": is_demo,
        "video": {"filename": "source.mp4"},
        "outputs": {
            "spectrum_plot": "spectrum.png",
            "displacement_plot": "displacement.png",
            "magnified_video": "magnified.mp4",
            "vibration_map": "vibration_map.png",
            "sound": "sound.wav",
        },
        "traffic": {"response_plot": "traffic-response.png"},
    }
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    for filename in (
        "source.mp4", "device_motion.json", "displacement.npy", "spectrum.png",
        "displacement.png", "magnified.mp4", "vibration_map.png", "sound.wav",
        "traffic-response.png",
    ):
        (folder / filename).write_bytes(filename.encode("utf-8"))
    return folder


def test_publish_scan_copies_only_result_and_declared_outputs(tmp_path):
    source = _scan_folder(tmp_path)
    copied = publish_results.publish_scan(source, "bridge-real", root=tmp_path)
    destination = tmp_path / "demo_scans" / "bridge-real"

    assert (destination / "result.json").is_file()
    assert {path.name for path in destination.iterdir()} == {
        "result.json", "spectrum.png", "displacement.png", "magnified.mp4",
        "vibration_map.png", "sound.wav", "traffic-response.png",
    }
    assert not (destination / "source.mp4").exists()
    assert not (destination / "device_motion.json").exists()
    assert not (destination / "displacement.npy").exists()
    assert len(copied) == 7


@pytest.mark.parametrize(
    ("status", "is_demo", "message"),
    [
        ("complete", True, "Demo results"),
        ("running", False, "completed scans"),
    ],
)
def test_publish_scan_refuses_demo_or_incomplete_result(tmp_path, status, is_demo, message):
    source = _scan_folder(tmp_path, status=status, is_demo=is_demo)
    with pytest.raises(ValueError, match=message):
        publish_results.publish_scan(source, "not-published", root=tmp_path)
    assert not (tmp_path / "demo_scans" / "not-published").exists()


def test_publish_scan_cli_prints_exact_copied_paths(tmp_path, monkeypatch, capsys):
    source = _scan_folder(tmp_path)
    monkeypatch.setattr(publish_results, "ROOT", tmp_path)
    assert publish_results.main(["--scan", str(source), "--name", "bridge-real"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "demo_scans/bridge-real/result.json",
        "demo_scans/bridge-real/spectrum.png",
        "demo_scans/bridge-real/displacement.png",
        "demo_scans/bridge-real/magnified.mp4",
        "demo_scans/bridge-real/vibration_map.png",
        "demo_scans/bridge-real/sound.wav",
        "demo_scans/bridge-real/traffic-response.png",
    ]


def test_publish_validation_creates_file_and_updates_index_without_duplicates(tmp_path):
    validation_dir = tmp_path / "frontend" / "public" / "validation"
    validation_dir.mkdir(parents=True)
    (validation_dir / "index.json").write_text(
        json.dumps(["existing.json", "proof.json", "proof.json"]), encoding="utf-8",
    )
    source = tmp_path / "comparison.json"
    payload = {"camera_frequency_hz": 3.7, "agreement_verdict": "PASS"}
    source.write_text(json.dumps(payload), encoding="utf-8")

    copied = publish_results.publish_validation(source, "proof", root=tmp_path)

    destination = validation_dir / "proof.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert json.loads((validation_dir / "index.json").read_text(encoding="utf-8")) == [
        "existing.json", "proof.json",
    ]
    assert copied == [
        Path("frontend/public/validation/proof.json"),
        Path("frontend/public/validation/index.json"),
    ]


def test_publish_validation_cli_and_list_show_both_publication_types(tmp_path, monkeypatch, capsys):
    source_scan = _scan_folder(tmp_path)
    publish_results.publish_scan(source_scan, "bridge-real", root=tmp_path)
    source = tmp_path / "comparison.json"
    source.write_text(json.dumps({"agreement_verdict": "PASS"}), encoding="utf-8")
    monkeypatch.setattr(publish_results, "ROOT", tmp_path)

    assert publish_results.main(["--validation", str(source), "--name", "proof"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "frontend/public/validation/proof.json",
        "frontend/public/validation/index.json",
    ]
    assert publish_results.main(["--list"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Showcase scans:", "  bridge-real", "Validation results:", "  proof",
    ]


def test_publish_rejects_unsafe_names_and_missing_output_files(tmp_path):
    source = _scan_folder(tmp_path)
    with pytest.raises(ValueError, match="NAME"):
        publish_results.publish_scan(source, "../escape", root=tmp_path)

    (source / "sound.wav").unlink()
    with pytest.raises(ValueError, match="missing"):
        publish_results.publish_scan(source, "broken", root=tmp_path)
    assert not (tmp_path / "demo_scans" / "broken").exists()
