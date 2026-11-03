import cv2
import numpy as np

from setu.meta_client import MetaClient
from setu.traffic import track_traffic


def _video(path, seconds=1.2, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    for _ in range(int(seconds * fps)):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()


def test_track_traffic_mock_fixture_response_and_plot(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    _video(source)
    client = MetaClient(mode="mock")
    prompts = []

    def track_video(_path, prompt, **kwargs):
        prompts.append(prompt)
        # Ten frames make a single object visible for one second.
        text = "\n".join(
            f"<{frame}f>0<|box;x1=1;y1=2;x2=20;y2=30;w=160;h=120|><|mask;x=0;y=0;data=1,1,!|>"
            for frame in range(10)
        ) if prompt == "bus" else ""
        return {"status": "completed", "output_text": text, "_mock_data": True}

    monkeypatch.setattr(client, "track_video", track_video)
    monkeypatch.setattr("setu.traffic.get_meta_client", lambda: client)
    plot = tmp_path / "traffic.png"
    result = track_traffic(source, displacement=np.arange(12), fps=10, out_png=plot)
    assert prompts == ["truck", "bus", "car"]
    assert result["events"] == [{"t_start": 0.0, "t_end": 1.0, "type": "bus"}]
    assert plot.is_file() and plot.stat().st_size > 0


def test_track_traffic_drops_short_detections(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    _video(source)
    client = MetaClient(mode="mock")
    response = {"output_text": "<0f>0<|box;x1=1;y1=1;x2=2;y2=2;w=160;h=120|>"}
    monkeypatch.setattr(client, "track_video", lambda *_a, **_kw: response)
    monkeypatch.setattr("setu.traffic.get_meta_client", lambda: client)
    assert track_traffic(source)["events"] == []


def test_public_demo_returns_reason_without_tracking(monkeypatch):
    class Client:
        mode = "live"
        api_key = "configured"
        def track_video(self, *_args, **_kwargs):
            raise AssertionError("public demo must not call tracking")

    monkeypatch.setattr("setu.traffic.get_meta_client", lambda: Client())
    monkeypatch.setenv("SETU_PUBLIC_DEMO", "true")
    result = track_traffic("unused.mp4")
    assert result["events"] == []
    assert "public demo" in result["reason"].lower()
