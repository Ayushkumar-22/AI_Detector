"""
app.py — AI Image Detector
Detects whether an uploaded image is AI-generated or a real photograph
using CLIP model embeddings from HuggingFace
"""
import streamlit as st
import sys
import os
import io
import json
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from config import settings

# ── Page config — must be first Streamlit call ─────────────────
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(99,179,237,0.05) 0%, transparent 70%);
        pointer-events: none;
    }

    .main-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        color: #63b3ed;
        margin: 0;
        letter-spacing: -1px;
    }

    .main-header p {
        color: rgba(255,255,255,0.5);
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }

    .result-ai {
        background: linear-gradient(135deg, #2d1515 0%, #1a0a0a 100%);
        border: 2px solid #fc8181;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-real {
        background: linear-gradient(135deg, #0f2d1a 0%, #0a1a0f 100%);
        border: 2px solid #68d391;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }

    .result-uncertain {
        background: linear-gradient(135deg, #2d2510 0%, #1a1508 100%);
        border: 2px solid #f6e05e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }

    .verdict-text {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.3rem 0;
    }

    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #63b3ed;
    }

    .metric-label {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    .artifact-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .prompt-card {
        background: rgba(99,179,237,0.05);
        border: 1px solid rgba(99,179,237,0.15);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        color: rgba(255,255,255,0.7);
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.88rem;
    }

    .stat-label { color: rgba(255,255,255,0.4); }
    .stat-value { color: rgba(255,255,255,0.85); font-family: 'Space Mono', monospace; }

    .badge-ai {
        background: rgba(252,129,129,0.15);
        border: 1px solid rgba(252,129,129,0.4);
        color: #fc8181;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .badge-real {
        background: rgba(104,211,145,0.15);
        border: 1px solid rgba(104,211,145,0.4);
        color: #68d391;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .progress-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
        margin: 0.3rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2b6cb0 0%, #2c5282 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        cursor: pointer;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #3182ce 0%, #2b6cb0 100%);
        transform: translateY(-1px);
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(99,179,237,0.3);
        border-radius: 12px;
        padding: 1rem;
        background: rgba(99,179,237,0.02);
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ───────────────────────────────────────────────
def init_state():
    defaults = {
        "result": None,
        "image_stats": None,
        "ela_score": None,
        "ela_image": None,
        "freq_analysis": None,
        "history": [],
        "processing_time": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Lazy loaders ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    from utils.detector import detector
    with st.spinner("⏳ Loading CLIP model... (first load takes 1-2 minutes)"):
        detector.load()
    return detector


# ── Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 AI Image Detector</h1>
    <p>CLIP-based detection · Real vs AI-Generated · Forensic Analysis</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    settings.DETECTION_THRESHOLD = 0.9

    use_ensemble = st.checkbox(
        "Use Ensemble (2 models)",
        value=True,
        help="Runs two CLIP models and averages results. More accurate but slower.",
        disabled=True,
    )
    settings.USE_ENSEMBLE = True

    run_forensics = st.checkbox(
        "Run Forensic Analysis",
        value=True,
        help="Runs ELA and frequency analysis alongside CLIP detection.",
        disabled=True,
    )

    st.markdown("---")
    st.markdown("## 📊 Session Stats")

    total = len(st.session_state.history)
    ai_count = sum(1 for h in st.session_state.history if h.get("is_ai"))
    real_count = total - ai_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", total)
    with col2:
        st.metric("🤖 AI", ai_count)
    with col3:
        st.metric("📷 Real", real_count)

    if st.session_state.history:
        st.markdown("**Recent:**")
        for h in reversed(st.session_state.history[-5:]):
            label = "🤖 AI" if h["is_ai"] else "📷 Real"
            conf = h["confidence"]
            st.caption(f"{label} · {conf:.0%} conf · {h['name'][:20]}")

    st.markdown("---")

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.session_state.result = None
        st.rerun()


# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════

col_upload, col_results = st.columns([1, 1], gap="large")

# ── LEFT: Upload ────────────────────────────────────────────────
with col_upload:
    st.markdown("### 📤 Upload Image")

    uploaded = st.file_uploader(
        "Upload an image to analyze",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image_bytes = uploaded.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        from utils.image_utils import resize_for_display
        display_img = resize_for_display(image, max_size=600)
        st.image(display_img, use_column_width=True, caption=uploaded.name)

        st.markdown("---")

        analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

        if analyze_btn:
            start_time = time.time()

            with st.spinner("🧠 Running CLIP analysis..."):
                try:
                    det = load_detector()
                    result = det.predict(image)
                    st.session_state.result = result

                    if run_forensics:
                        from utils.image_utils import get_image_stats, compute_ela_score, get_frequency_analysis
                        with st.spinner("🔬 Running forensic analysis..."):
                            st.session_state.image_stats = get_image_stats(image)
                            ela_score, ela_img = compute_ela_score(image)
                            st.session_state.ela_score = ela_score
                            st.session_state.ela_image = ela_img
                            st.session_state.freq_analysis = get_frequency_analysis(image)

                    elapsed = time.time() - start_time
                    st.session_state.processing_time = elapsed

                    # Add to history
                    st.session_state.history.append({
                        "name": uploaded.name,
                        "is_ai": result["is_ai_generated"],
                        "confidence": result["confidence"],
                        "ai_prob": result["ai_probability"],
                    })

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    if settings.DEBUG_MODE:
                        import traceback
                        st.code(traceback.format_exc())

    else:
        st.info("👆 Upload a JPG, PNG, or WEBP image to begin analysis.")

        st.markdown("**What this tool detects:**")
        items = [
            "Midjourney generated images",
            "DALL-E generated images",
            "Stable Diffusion outputs",
            "GAN-generated faces",
            "Deepfakes",
            "Any AI synthesized imagery",
        ]
        for item in items:
            st.markdown(f"- {item}")


# ── RIGHT: Results ───────────────────────────────────────────────
with col_results:
    st.markdown("### 📊 Analysis Results")

    result = st.session_state.result

    if result is None:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
        border-radius:12px; padding:3rem; text-align:center; color:rgba(255,255,255,0.3);">
            <div style="font-size:3rem;">🔍</div>
            <div style="margin-top:1rem;">Upload an image and click Analyze to see results</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        is_ai = result["is_ai_generated"]
        ai_prob = result["ai_probability"]
        real_prob = result["real_probability"]
        confidence = result["confidence"]

        # ── VERDICT CARD ─────────────────────────────────────
        is_uncertain = result.get("is_uncertain", False)

        if is_uncertain:
            verdict_class = "result-uncertain"
            verdict_icon = "⚠️"
            verdict_text = "UNCERTAIN"
            verdict_color = "#f6e05e"
        elif is_ai:
            verdict_class = "result-ai"
            verdict_icon = "🤖"
            verdict_text = "AI GENERATED"
            verdict_color = "#fc8181"
        else:
            verdict_class = "result-real"
            verdict_icon = "📷"
            verdict_text = "REAL PHOTO"
            verdict_color = "#68d391"

        st.markdown(f"""
        <div class="{verdict_class}">
            <div style="font-size:3rem;">{verdict_icon}</div>
            <div class="verdict-text" style="color:{verdict_color};">{verdict_text}</div>
            <div style="color:rgba(255,255,255,0.5); font-size:0.85rem; margin-top:0.3rem;">
                Confidence: {confidence:.1%} · Processed in {st.session_state.processing_time:.1f}s
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── PROBABILITY BARS ──────────────────────────────────
        st.markdown("#### Probability Breakdown")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#fc8181;">{ai_prob:.1%}</div>
                <div class="metric-label">AI Generated</div>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#68d391;">{real_prob:.1%}</div>
                <div class="metric-label">Real Photo</div>
            </div>
            """, unsafe_allow_html=True)

        # Visual bars
        ai_pct = int(ai_prob * 100)
        real_pct = int(real_prob * 100)
        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:rgba(255,255,255,0.4); margin-bottom:4px;">
                <span>🤖 AI</span><span>{ai_pct}%</span>
            </div>
            <div class="progress-bar-container">
                <div style="width:{ai_pct}%; height:100%; background:linear-gradient(90deg,#fc8181,#e53e3e); border-radius:6px;"></div>
            </div>
        </div>
        <div style="margin: 0.5rem 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:rgba(255,255,255,0.4); margin-bottom:4px;">
                <span>📷 Real</span><span>{real_pct}%</span>
            </div>
            <div class="progress-bar-container">
                <div style="width:{real_pct}%; height:100%; background:linear-gradient(90deg,#68d391,#38a169); border-radius:6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── TABS ──────────────────────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Artifacts", "📡 Forensics", "🧩 CLIP Details", "📄 Raw Data"
        ])

        # TAB 1: Artifact Analysis
        with tab1:
            st.markdown("#### Visual Artifact Analysis")
            st.caption("Per-feature comparison: real vs AI-like characteristics")

            artifact_data = result.get("artifact_analysis", {})
            artifact_labels = {
                "skin_texture": "Skin Texture",
                "background": "Background",
                "lighting": "Lighting",
                "edges": "Edge Quality",
                "symmetry": "Face Symmetry",
            }

            suspicious_count = 0
            for key, label in artifact_labels.items():
                if key not in artifact_data:
                    continue
                a = artifact_data[key]
                suspicious = a["suspicious"]
                score = a["suspicion_score"]
                pct = int(score * 100)

                if suspicious:
                    suspicious_count += 1
                    badge = '<span class="badge-ai">⚠ Suspicious</span>'
                    bar_color = "#fc8181"
                else:
                    badge = '<span class="badge-real">✓ Normal</span>'
                    bar_color = "#68d391"

                st.markdown(f"""
                <div class="artifact-card">
                    <div>
                        <div style="font-weight:500; font-size:0.9rem;">{label}</div>
                        <div style="font-size:0.75rem; color:rgba(255,255,255,0.35); margin-top:2px;">
                            Suspicion score: {score:.2f}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        {badge}
                        <div class="progress-bar-container" style="width:80px; margin-top:4px; margin-left:auto;">
                            <div style="width:{pct}%; height:100%; background:{bar_color}; border-radius:6px;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            total_features = len(artifact_labels)
            st.markdown(f"""
            <div style="margin-top:1rem; padding:0.8rem; background:rgba(255,255,255,0.02);
            border-radius:8px; text-align:center; font-size:0.85rem; color:rgba(255,255,255,0.5);">
                {suspicious_count}/{total_features} features flagged as suspicious
            </div>
            """, unsafe_allow_html=True)


        # TAB 2: Forensic Analysis
        with tab2:
            if st.session_state.image_stats:
                st.markdown("#### Image Statistics")
                stats = st.session_state.image_stats

                stat_items = [
                    ("Dimensions", f"{stats['width']} × {stats['height']} px"),
                    ("Megapixels", f"{stats['megapixels']} MP"),
                    ("Aspect Ratio", str(stats['aspect_ratio'])),
                    ("Mean Brightness", str(stats['mean_brightness'])),
                    ("Sharpness Score", str(stats['sharpness_score'])),
                    ("Noise Level", str(stats['noise_level'])),
                    ("Color Std (R/G/B)", f"{stats['color_std_r']} / {stats['color_std_g']} / {stats['color_std_b']}"),
                ]

                for label, value in stat_items:
                    st.markdown(f"""
                    <div class="stat-row">
                        <span class="stat-label">{label}</span>
                        <span class="stat-value">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)

                if st.session_state.ela_score is not None:
                    st.markdown("#### Error Level Analysis (ELA)")
                    st.caption("Measures compression artifacts. Uniform ELA = possible AI generation.")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{st.session_state.ela_score:.4f}</div>
                        <div class="metric-label">ELA Score (higher = more natural variation)</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.session_state.ela_image:
                        st.image(
                            st.session_state.ela_image,
                            caption="ELA Visualization (bright areas = high error = compression artifacts)",
                            use_column_width=True,
                        )

                if st.session_state.freq_analysis:
                    st.markdown("#### Frequency Analysis (FFT)")
                    freq = st.session_state.freq_analysis
                    f_items = [
                        ("Low Freq Energy", str(freq["low_frequency_energy"])),
                        ("High Freq Energy", str(freq["high_frequency_energy"])),
                        ("Freq Ratio", str(freq["freq_ratio"])),
                    ]
                    for label, value in f_items:
                        st.markdown(f"""
                        <div class="stat-row">
                            <span class="stat-label">{label}</span>
                            <span class="stat-value">{value}</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Enable 'Run Forensic Analysis' in the sidebar to see this data.")


        # TAB 3: CLIP Details
        with tab3:
            st.markdown("#### Similarity Scores")
            st.caption("How similar the image is to each text prompt")

            col_r, col_ai = st.columns(2)

            with col_r:
                st.markdown("**📷 Real Image Prompts**")
                from utils.detector import REAL_IMAGE_PROMPTS
                real_scores = result.get("real_scores", [])
                if real_scores:
                    sorted_real = sorted(
                        zip(real_scores, REAL_IMAGE_PROMPTS),
                        reverse=True
                    )
                    for score, prompt in sorted_real[:5]:
                        pct = int(((score + 1) / 2) * 100)
                        st.markdown(f"""
                        <div class="prompt-card">
                            <div style="font-size:0.75rem; color:rgba(255,255,255,0.4);">{score:.4f}</div>
                            <div>{prompt[:55]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

            with col_ai:
                st.markdown("**🤖 AI Image Prompts**")
                from utils.detector import AI_GENERATED_PROMPTS
                ai_scores = result.get("ai_scores", [])
                if ai_scores:
                    sorted_ai = sorted(
                        zip(ai_scores, AI_GENERATED_PROMPTS),
                        reverse=True
                    )
                    for score, prompt in sorted_ai[:5]:
                        st.markdown(f"""
                        <div class="prompt-card" style="border-color:rgba(252,129,129,0.15);">
                            <div style="font-size:0.75rem; color:rgba(255,255,255,0.4);">{score:.4f}</div>
                            <div>{prompt[:55]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("#### Top Matching Prompts")
            st.success(f"📷 Best real match: *{result.get('top_real_prompt', '')}*")
            st.error(f"🤖 Best AI match: *{result.get('top_ai_prompt', '')}*")

            if result.get("ensemble"):
                st.markdown("#### Ensemble Model Results")
                m1 = result.get("model1_result", {})
                m2 = result.get("model2_result", {})
                ecol1, ecol2 = st.columns(2)
                with ecol1:
                    st.markdown(f"**Model 1:** `{settings.CLIP_MODEL.split('/')[-1]}`")
                    st.metric("AI Probability", f"{m1.get('ai_probability', 0):.1%}")
                with ecol2:
                    st.markdown(f"**Model 2:** `{settings.CLIP_MODEL_2.split('/')[-1]}`")
                    st.metric("AI Probability", f"{m2.get('ai_probability', 0):.1%}")


        # TAB 4: Raw Data
        with tab4:
            st.markdown("#### Full Detection Output")
            # Remove nested model results for cleaner display
            clean_result = {k: v for k, v in result.items()
                          if k not in ["model1_result", "model2_result"]}
            st.json(clean_result)

            # Download button
            result_json = json.dumps(result, indent=2)
            st.download_button(
                label="⬇️ Download Full Report (JSON)",
                data=result_json,
                file_name=f"detection_report.json",
                mime="application/json",
            )


# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:rgba(255,255,255,0.2); font-size:0.78rem; padding:1rem 0;">
    AI Image Detector · CLIP Model · HuggingFace Transformers · For Research Use Only
    <br>Results are probabilistic — not 100% accurate. Always verify with multiple methods.
</div>
""", unsafe_allow_html=True)
