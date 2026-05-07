"""
test_setup.py — verify all components work before running the app
Run: python test_setup.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 55)
print("AI Image Detector — Setup Verification")
print("=" * 55)

# Test 1: Config
print("\n[1] Testing config...")
try:
    from config import settings
    print(f"   ✅ Config loaded")
    print(f"       Model: {settings.CLIP_MODEL}")
    print(f"       Ensemble: {settings.USE_ENSEMBLE}")
    print(f"       Threshold: {settings.DETECTION_THRESHOLD}")
except Exception as e:
    print(f"   ❌ Config failed: {e}")

# Test 2: PIL / Image utils
print("\n[2] Testing image utilities...")
try:
    from PIL import Image
    import numpy as np
    from utils.image_utils import get_image_stats, compute_ela_score

    # Create a dummy test image
    dummy = Image.fromarray(
        (np.random.rand(256, 256, 3) * 255).astype("uint8")
    )
    stats = get_image_stats(dummy)
    ela_score, _ = compute_ela_score(dummy)
    print(f"   ✅ Image utils working")
    print(f"       Stats keys: {list(stats.keys())[:4]}...")
    print(f"       ELA score: {ela_score}")
except Exception as e:
    print(f"   ❌ Image utils failed: {e}")

# Test 3: PyTorch
print("\n[3] Testing PyTorch...")
try:
    import torch
    print(f"   ✅ PyTorch {torch.__version__}")
    print(f"       CUDA available: {torch.cuda.is_available()}")
    print(f"       Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    print(f"   ❌ PyTorch failed: {e}")

# Test 4: Transformers / CLIP imports
print("\n[4] Testing HuggingFace Transformers...")
try:
    from transformers import CLIPModel, CLIPProcessor
    print(f"   ✅ Transformers imported successfully")
except Exception as e:
    print(f"   ❌ Transformers failed: {e}")

# Test 5: Detector import (no model download yet)
print("\n[5] Testing detector module...")
try:
    from utils.detector import CLIPDetector, EnsembleDetector, REAL_IMAGE_PROMPTS, AI_GENERATED_PROMPTS
    print(f"   ✅ Detector module imported")
    print(f"       Real prompts: {len(REAL_IMAGE_PROMPTS)}")
    print(f"       AI prompts: {len(AI_GENERATED_PROMPTS)}")
except Exception as e:
    print(f"   ❌ Detector failed: {e}")

# Test 6: Streamlit
print("\n[6] Testing Streamlit...")
try:
    import streamlit
    print(f"   ✅ Streamlit {streamlit.__version__}")
except Exception as e:
    print(f"   ❌ Streamlit failed: {e}")

print("\n" + "=" * 55)
print("Setup verification complete!")
print("If all checks passed, run: streamlit run app.py")
print("=" * 55)
