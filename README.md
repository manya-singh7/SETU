# SETU: Structural Evaluation and Telemetry Utility

SETU is a vision-based screening tool that measures natural structural vibration frequencies of civil infrastructure from smartphone video recordings. It tracks subpixel displacements of structural elements relative to a static background to compute power spectral density and extract resonant modal peaks.

## Core Principle

SETU never declares a bridge safe. It flags readings that need human civil engineer review and on-site physical inspection. Weak or failed verdicts are not anomalies.

## How It Works

The processing pipeline executes in seven sequential stages:

1. **Video Checks** (`setu/video_checks.py`): Validates file container integrity, decodability, duration (5 to 60 seconds), frame rate (15 to 240 fps), and minimum resolution (320x240 pixels).
2. **SAM 3.1 Segmentation** (`setu/segment.py`): Segments the vibrating structural target and a static background reference anchor on Frame 0 using natural language text prompts.
3. **Background Stabilisation** (`setu/motion.py`): Tracks subpixel motion in the static background anchor to isolate and remove camera micro-jitter and drift.
4. **Motion Tracking** (`setu/motion.py`): Tracks subpixel displacement of the structural target across all frames using phase correlation or Lucas-Kanade optical flow.
5. **Spectrum Analysis** (`setu/spectrum.py`): Computes Welch Power Spectral Density (PSD), extracts resonant frequency peaks, checks peak prominence and signal-to-noise ratio, calculates 95% confidence intervals, and assigns a reliability verdict (usable, weak, or failed).
6. **Muse Spark Interpretation** (`setu/interpret.py`, `setu/assess.py`, `setu/quality_gate.py`): Runs only when the spectrum verdict is usable. Queries OpenStreetMap for bridge metadata, checks visual quality gates, and produces an evidence-locked anomaly assessment. Weak or failed scans skip this stage.
7. **Optional Magnification** (`setu/evm.py`): Applies phase-based Eulerian video magnification centered at the dominant frequency to generate an amplified video for visual inspection display only.

## Meta Models Used

SETU integrates three Meta model lines:

- **sam-3.1** (`setu/segment.py`): Used for Promptable Concept Segmentation (PCS) on Frame 0. It takes user prompts (such as "concrete pillar" and "building") to generate binary target and background masks. An offline geometric heuristic segmenter acts as a fallback when the Meta API is unavailable.
- **muse-spark-1.3** (`setu/assess.py`, `setu/quality_gate.py`, `setu/interpret.py`): Used for LLM-assisted anomaly screening and visual quality gate verification. It operates under a strict evidence lock, meaning every number in the output must match values in the input telemetry or OpenStreetMap metadata. It never certifies safety and always recommends human inspection when anomalies or uncertainties occur.
- **muse-voice-transcribe-1.0** (`setu/voice.py`): Transcribes short driver voice notes; Muse Spark turns transcripts into engineer tags with exact transcript evidence.

## Public Demo Deployment

The root `Dockerfile` builds the frontend and serves it with the FastAPI backend
on one port. It includes ffmpeg and PyAV support. Set
`SETU_PUBLIC_DEMO=true` for public hosting: Meta runs in cached mode with live
requests blocked, uploads are limited to 3 per IP per hour and 60 seconds, and
original uploaded videos are deleted after 24 hours. See
[`deploy/README.md`](deploy/README.md) for beginner steps to deploy on Modal or
Render.

Real showcase scans can be bundled under `demo_scans/<scan-name>/` with a
`result.json` and referenced output files. They are listed at `GET /api/showcase`
and served read-only.

## Setup on Windows

### Prerequisites

- Python 3.11 or 3.12
- Node.js 18 or 20 (LTS)

### Installation

1. Clone or open the repository in PowerShell.
2. Install Python dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Install frontend dependencies:
   ```powershell
   npm install --prefix frontend
   ```
4. Copy the environment configuration template:
   ```powershell
   copy .env.example .env
   ```

### Running the Application

To start both the FastAPI backend (port 8000) and the Vite frontend (port 5173) in separate windows:
```powershell
.\scripts\start_dev.ps1
```

### Demo Mode (No API Keys Required)

To run the application entirely offline without Meta or cloud API keys:
```powershell
.\scripts\start_dev.ps1 -Demo
```
Alternatively, set the environment variable before starting:
```powershell
$env:SETU_DEMO_MODE = "true"
.\scripts\start_dev.ps1
```
In demo mode, uploads skip live model inference and return validated benchmark fixtures.

## Command Line Usage

Run the screening pipeline directly on a video file:
```powershell
python -m setu.run path\to\video.mp4 --target "corner of concrete pillar" --background "stationary ground reference"
```

Optional manual bounding box overrides can be provided:
```powershell
python -m setu.run path\to\video.mp4 --target "pillar" --background "ground" --bbox-target 270,90,100,280 --bbox-background 40,360,120,80
```

The CLI outputs:
- Video properties (resolution, frame rate, duration)
- Segmentation method (SAM 3.1 or fallback)
- Subpixel tracking metrics and background residual motion
- Modal frequency peaks with relative power and SNR
- Quality verdict and warnings
- Saved displacement and power spectral density plots (`*_displacement.png`, `*_spectrum.png`)

## Running Tests

Run the fast test suite (excluding long-running video magnification benchmarks):
```powershell
python -m pytest tests -m "not slow"
```

Run the complete test suite including end-to-end video synthesis:
```powershell
python -m pytest
```

Run specific test modules:
```powershell
python -m pytest tests/test_api.py -v
python -m pytest tests/test_integration.py -v
```

## Validation Tool (Phyphox Cross-Sensor Comparison)

SETU includes a cross-sensor validation module in `validation/compare.py` to compare optical camera measurements against contact accelerometer data exported from the mobile app phyphox (https://phyphox.org).

### Usage

Compare a phyphox CSV export against a known camera frequency:
```powershell
python -m validation.compare --phyphox tests/fixtures/phyphox_sample.csv --freq 3.70 --out validation_plot.png
```

Compare a phyphox CSV export against a SETU scan result JSON:
```powershell
python -m validation.compare --phyphox tests/fixtures/phyphox_sample.csv --result data/scans/<scan_id>/result.json --out validation_plot.png
```

### What It Reports

- Detected accelerometer recording axes and sampling rate
- Dominant vibration axis (X, Y, or Z) based on spectral peak power
- Accelerometer resonant peak frequency with 95% confidence interval
- Camera resonant peak frequency with confidence interval
- Absolute frequency difference in Hz and relative difference in percent
- Combined uncertainty calculation
- Agreement is judged by combined uncertainty from the camera confidence interval and accelerometer confidence interval.
- Verdict: PASS or FAIL based on whether the measured frequency intervals agree within their combined uncertainty.
- Composite dual-spectrum plot normalizing both optical and contact sensor PSDs

## Limitations

- **Frequency only, not damage detection**: Natural frequency shifts indicate changes in structural stiffness or mass, but frequency data alone cannot detect crack locations, rebar corrosion, fatigue, foundation scour, or load capacity.
- **Pixel displacement units**: Displacement is measured in image pixels. Translating pixel motion to millimeters requires a metric scale factor or known ground-truth dimensions.
- **Synthetic calibration**: Reliability thresholds and signal-to-noise criteria were tuned on synthetic video sequences and are being recalibrated against real bridge recordings.
- **Mounting and background requirements**: The recording device must be braced against a solid rest or mounted on a tripod. The frame must contain a stationary reference object (bedrock, distant building, ground) to cancel camera micro-tremor.
- **Environmental variations**: Daily ambient temperature shifts and dynamic traffic loads alter natural frequencies independently of structural damage. Baseline comparisons must account for these seasonal and thermal cycles.

## Results

<!--
TODO: Placeholders for real-footage validation results.
Do not insert simulated or placeholder numbers in this section.
Real benchmark numbers will be populated from field recording sessions.
-->

### Field Test Results (Real Footage)

- **Structure**: [TODO: Bridge Name / Location]
- **Sensor Setup**: [TODO: Camera model, resolution, frame rate, distance, bracing method]
- **Contact Accelerometer Reference**: [TODO: Phyphox export file reference]
- **Measured Camera Frequency (Hz)**: [TODO]
- **Measured Accelerometer Frequency (Hz)**: [TODO]
- **Absolute Discrepancy (Hz)**: [TODO]
- **Relative Discrepancy (%)**: [TODO]
- **Validation Verdict**: [TODO]
- **Notes / Observations**: [TODO]
