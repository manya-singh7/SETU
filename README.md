# SETU: Structural Evaluation & Telemetry Utility

> **Hackathon Edition** — Vision-based civil infrastructure screening via mobile phone video, Eulerian motion amplification, and AI-assisted triage.

---

## ⚠️ Core Safety Principle

> **SETU NEVER DECLARES A BRIDGE SAFE.**
>
> Mobile video vibration estimation is an **early screening heuristic only**. It isolates structural vibration frequencies to flag anomalies, stiffness shifts, or deviations from nominal engineering bounds. **Every flagged reading is routed to a human civil engineering triage queue for on-site physical inspection.**

---

## The 7-Stage Pipeline

```
[1. 30s Phone Video + GPS]
           │
           ▼
[2. Meta SAM 3.1 Segmentation]
   ├── Text prompt target (e.g. "corner of concrete pillar")
   └── Stationary background reference (e.g. "bedrock / distant building")
           │
           ▼
[3. Subpixel Displacement Tracking]
   └── Corrected Motion = Target Motion - Stationary Background Motion
           │
           ▼
[4. Welch PSD / FFT Frequency Extraction]
   └── Natural modal frequencies (f₁, f₂, f₃) & half-power damping
           │
           ▼
[5. Eulerian Video Magnification (EVM)]
   └── Motion-amplified video produced FOR VISUAL INSPECTION DISPLAY ONLY
           │
           ▼
[6. OpenStreetMap + Meta Muse Spark 1.3]
   ├── Overpass API queries bridge type, material, and span length
   └── Muse Spark 1.3 formulates a cautious, non-authoritative triage report
           │
           ▼
[7. Regional Map & Civil Engineer Triage Queue]
   └── Leaflet map markers categorized by severity (UNUSUAL, REQUIRES_INSPECTION, INCONCLUSIVE)
```

---

## API Research & Swappable Interfaces

To prevent hard dependencies on external cloud APIs or GPU hardware during local hackathon demos, both external models are placed behind clean abstract interfaces with local swappable fallbacks:

### 1. Meta SAM 3.1 (`setu/segment.py`)
- **Model:** Meta SAM 3.1 (`facebook/sam3.1`)
- **Capability:** Promptable Concept Segmentation (PCS) with open-vocabulary text cues (e.g., `"corner of concrete pillar"`) and Object Multiplex joint tracking across video frames.
- **Interface:** `BaseSegmenter` (`segment(frame, target_prompt, background_prompt)`)
- **Implementations:**
  - `MetaSAM31Segmenter`: Meta Model API integration.
  - `FallbackSegmenter`: Offline-friendly geometric ROI segmenter for zero-latency local development.

### 2. Meta Muse Spark 1.3 (`setu/assess.py`)
- **Model:** Meta Muse Spark 1.3 (`muse-spark-1.3`)
- **Endpoint:** Meta Model API (`https://api.meta.ai/v1`), using standard OpenAI-compatible client libraries.
- **System Prompt:** Enforces the strict rule that it must **never certify a structure safe** and must always conclude with an on-site inspection directive.
- **Interface:** `BaseAssessor` (`assess(bridge, spectrum)`)
- **Implementations:**
  - `MuseSparkAssessor`: Meta Model API reasoning client.
  - `LocalRuleBasedAssessor`: Deterministic physics-based heuristic comparing fundamental frequency against nominal $f_1 \approx 100/L$ span benchmarks.

---

## Repository Structure

```
SETU/
├── setu/                        # Modular backend pipeline package
│   ├── __init__.py              # Package exports & versioning
│   ├── config.py                # Pydantic-settings loading from .env (no secrets in code)
│   ├── models.py                # SQLModel database tables (Bridge, VideoUpload, AnalysisResult, TriageItem)
│   ├── database.py              # SQLite engine & session factory
│   ├── segment.py               # Step 2: SAM 3.1 Promptable Concept Segmentation & Fallback
│   ├── motion.py                # Step 3: Subpixel tracking & camera motion cancellation
│   ├── spectrum.py              # Step 4: Welch PSD & FFT dominant peak picking
│   ├── evm.py                   # Step 5: Eulerian Video Magnification (display only)
│   ├── osm.py                   # Step 6A: OpenStreetMap Overpass bridge query client
│   ├── assess.py                # Step 6B: Muse Spark 1.3 & Rule-based anomaly assessment
│   └── api.py                   # Step 7: FastAPI server & triage endpoints
│
├── tests/                       # Pytest verification suite
│   ├── conftest.py              # Database initialization fixtures
│   ├── test_api.py              # Endpoint tests (/health, /api/triage, /api/bridges)
│   ├── test_assess.py           # Safety philosophy & triage severity tests
│   └── test_spectrum.py         # Synthetic sinusoidal signal frequency tests
│
├── frontend/                    # Local React + Vite + Leaflet web application
│   ├── src/
│   │   ├── App.tsx              # Split dashboard: Leaflet map, triage queue, 30s video upload
│   │   ├── main.tsx             # React entry point with Leaflet CSS
│   │   ├── index.css            # Tailwind CSS styling
│   │   └── vite-env.d.ts        # TypeScript asset declarations
│   ├── package.json             # Vite, React 18, React-Leaflet, Lucide icons
│   └── vite.config.ts           # Vite dev server configuration (port 5173)
│
├── .env.example                 # Environment variable templates
├── .env                         # Local environment configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Language & Runtime** | Python 3.11+, Node.js 18+ |
| **Backend & API** | FastAPI, Uvicorn, SQLModel (SQLite), Pydantic v2 |
| **Signal & Video** | OpenCV (`opencv-python-headless`), NumPy, SciPy (Welch PSD, FFT) |
| **Foundation Models** | Meta SAM 3.1 (PCS Text Segmentation), Meta Muse Spark 1.3 (LLM Reasoning) |
| **Frontend** | React 18, Vite, Leaflet, React-Leaflet, Tailwind CSS, Lucide Icons |

---

## Quick Start (Run Locally)

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn setu.api:app --host 127.0.0.1 --port 8000 --reload
```

Verify backend health:
```bash
curl http://127.0.0.1:8000/health
# Response: {"status":"ok","service":"SETU Civil Telemetry API","fallbacks_enabled":true,"database":"sqlite:///./setu.db"}
```

Run test suite (including synthetic signal verification):
```bash
python -m pytest tests -v
```

### 2. Frontend Setup

```bash
cd frontend

# Install npm packages
npm install

# Start Vite local development server
npm run dev
```

Visit **`http://localhost:5173`** in your browser.

Build check:
```bash
npm run build
```

---

## Environment Variables (`.env`)

```ini
# Meta Model API (Optional if USE_LOCAL_FALLBACKS=true)
META_API_KEY=
META_API_BASE_URL=https://api.meta.ai/v1
MUSE_MODEL_NAME=muse-spark-1.3
SAM_MODEL_NAME=facebook/sam3.1

# Keep everything runnable locally out-of-the-box
USE_LOCAL_FALLBACKS=true

# Database Configuration (SQLite via SQLModel)
DATABASE_URL=sqlite:///./setu.db

# Server Configuration
HOST=127.0.0.1
PORT=8000
ENVIRONMENT=development
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# OpenStreetMap Overpass API
OVERPASS_API_URL=https://overpass-api.de/api/interpreter

# File Storage
STORAGE_DIR=./uploads
```
