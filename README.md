# AI Image Detector

Detects whether an image is AI-generated or a real photograph using OpenAI's CLIP model via HuggingFace Transformers.

---

## How It Works

CLIP (Contrastive Language-Image Pretraining) is a model that learns to match images with text descriptions. This tool exploits that capability by:

1. Encoding the uploaded image into an embedding vector using CLIP's vision encoder
2. Encoding two sets of text prompts — one describing real photographs, one describing AI-generated images — using CLIP's text encoder
3. Computing cosine similarity between the image and both prompt sets
4. The set with higher average similarity determines the verdict
5. Additional forensic analysis (ELA, frequency analysis, artifact detection) adds supporting evidence

---

## Project Structure

```
ai_detector/
├── app.py                  # Main Streamlit application
├── config.py               # Settings loader
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── test_setup.py           # Setup verification script
│
├── utils/
│   ├── __init__.py
│   ├── detector.py         # CLIP detection engine + prompts
│   └── image_utils.py      # ELA, frequency analysis, image stats
│
└── models/
    └── cache/              # Auto-created — CLIP model files stored here
```

---

## Prerequisites

- Python 3.9 or higher
- pip
- ~3 GB disk space (for CLIP model files)
- Internet connection (first run downloads models)

No API key required — models run fully locally via HuggingFace Transformers.

---

## Installation

### Step 1 — Clone or unzip the project

```bash
unzip ai_detector.zip
cd ai_detector
```

### Step 2 — Create virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, HuggingFace Transformers, Streamlit, and supporting libraries. Total download is approximately 1.5-2 GB on first install.

### Step 4 — Configure environment (optional)

```bash
cp .env.example .env
```

The defaults work fine without any changes. Edit `.env` only if you want to:
- Change which CLIP model is used
- Adjust the detection threshold
- Add a HuggingFace token for private models

### Step 5 — Verify setup

```bash
python test_setup.py
```

All six checks should show green checkmarks.

### Step 6 — Run the app

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

**First startup:** Downloads CLIP model files (~1.7 GB for large model). This takes 2-5 minutes depending on your connection. Subsequent startups are fast because models are cached in `./models/cache/`.

---

## Usage

1. Upload a JPG, PNG, or WEBP image using the file uploader
2. Click **Analyze Image**
3. View the verdict — AI Generated / Real Photo / Uncertain
4. Check the four result tabs:
   - **Artifacts** — per-feature analysis (skin, lighting, edges, symmetry)
   - **Forensics** — ELA visualization, image stats, frequency analysis
   - **CLIP Details** — similarity scores for all prompts
   - **Raw Data** — full JSON output, downloadable

### Settings (sidebar)

- **Detection Threshold** — 0.5 by default. Increase to 0.6-0.7 to reduce false positives
- **Use Ensemble** — runs two CLIP models for better accuracy (slower)
- **Run Forensic Analysis** — adds ELA and FFT analysis

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIP_MODEL` | `openai/clip-vit-large-patch14` | Primary CLIP model |
| `CLIP_MODEL_2` | `openai/clip-vit-base-patch32` | Secondary model for ensemble |
| `USE_ENSEMBLE` | `true` | Use both models |
| `DETECTION_THRESHOLD` | `0.5` | AI probability cutoff |
| `HF_TOKEN` | empty | HuggingFace token (optional) |
| `MODEL_CACHE_DIR` | `./models/cache` | Where to store models |

---

## Limitations

- CLIP was not specifically trained for AI detection — it uses zero-shot transfer
- Works best on photorealistic AI images (Midjourney, DALL-E, Stable Diffusion)
- May struggle with artistic styles, illustrations, or heavily edited real photos
- Accuracy is approximately 70-85% depending on image type
- Not a forensic tool — results are probabilistic

---

## Troubleshooting

**Slow first load**
Models download on first use (~1.7 GB). Subsequent runs load from cache.

**Out of memory error**
Switch to the smaller model: set `CLIP_MODEL=openai/clip-vit-base-patch32` in `.env`

**CUDA errors**
The app falls back to CPU automatically if CUDA is unavailable.

**`ModuleNotFoundError`**
Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`
