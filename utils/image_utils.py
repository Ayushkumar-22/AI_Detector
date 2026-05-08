"""
utils/image_utils.py — image preprocessing and analysis helpers
"""
import io
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat
from typing import Tuple, List


def load_image(image_bytes: bytes) -> Image.Image:
    """Load image from bytes, convert to RGB."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def jpeg_compress(image: Image.Image, quality: int = 85) -> Image.Image:
    """Re-encode the image at a lower JPEG quality to simulate compression."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_gaussian_blur(image: Image.Image, radius_range: Tuple[float, float] = (0.8, 2.5), probability: float = 0.8) -> Image.Image:
    """Apply gaussian blur randomly to simulate slight focus issues."""
    if np.random.rand() > probability:
        return image
    radius = float(np.random.uniform(*radius_range))
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def random_motion_blur(image: Image.Image, kernel_size_range: Tuple[int, int] = (3, 11), probability: float = 0.6) -> Image.Image:
    """Apply a simple motion blur kernel to simulate camera shake."""
    if np.random.rand() > probability:
        return image

    width, height = image.size
    max_kernel = min(max(kernel_size_range), width, height)
    if max_kernel < 3:
        return image

    possible_sizes = [s for s in range(kernel_size_range[0], kernel_size_range[1] + 1, 2) if s <= max_kernel]
    if not possible_sizes:
        return image

    kernel_size = int(np.random.choice(possible_sizes))
    kernel = [1.0 / kernel_size] * kernel_size
    direction = np.random.choice(["horizontal", "vertical"])

    try:
        if direction == "horizontal":
            return image.filter(ImageFilter.Kernel((kernel_size, 1), kernel, scale=1))
        return image.filter(ImageFilter.Kernel((1, kernel_size), kernel, scale=1))
    except ValueError:
        return image


def random_noise(image: Image.Image, std_range: Tuple[float, float] = (0.02, 0.10), probability: float = 0.7) -> Image.Image:
    """Inject low-level Gaussian noise to simulate sensor noise and compression artifacts."""
    if np.random.rand() > probability:
        return image
    arr = np.asarray(image).astype(np.float32) / 255.0
    std = float(np.random.uniform(*std_range))
    noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def random_brightness_contrast(image: Image.Image, brightness_range: Tuple[float, float] = (0.75, 1.25), contrast_range: Tuple[float, float] = (0.75, 1.25), probability: float = 0.8) -> Image.Image:
    """Randomly perturb brightness and contrast to simulate varying lighting conditions."""
    if np.random.rand() > probability:
        return image
    image = ImageEnhance.Brightness(image).enhance(float(np.random.uniform(*brightness_range)))
    image = ImageEnhance.Contrast(image).enhance(float(np.random.uniform(*contrast_range)))
    return image


def random_color_jitter(image: Image.Image, saturation_range: Tuple[float, float] = (0.8, 1.2), hue_shift: float = 0.05, probability: float = 0.6) -> Image.Image:
    """Randomly perturb color saturation and hue to simulate different camera processing."""
    if np.random.rand() > probability:
        return image
    image = ImageEnhance.Color(image).enhance(float(np.random.uniform(*saturation_range)))
    if np.random.rand() < 0.5:
        hsv = np.array(image.convert("HSV"), dtype=np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + int(np.random.uniform(-hue_shift * 255, hue_shift * 255))) % 256
        image = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
    return image


def random_crop_resize(image: Image.Image, scale_range: Tuple[float, float] = (0.7, 1.0), probability: float = 0.7) -> Image.Image:
    """Randomly crop and resize to simulate low-resolution or slightly framed real images."""
    if np.random.rand() > probability:
        return image
    w, h = image.size
    scale = float(np.random.uniform(*scale_range))
    crop_w, crop_h = int(w * scale), int(h * scale)
    if crop_w < 10 or crop_h < 10:
        return image
    left = np.random.randint(0, w - crop_w + 1)
    top = np.random.randint(0, h - crop_h + 1)
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    return cropped.resize((w, h), Image.LANCZOS)


def preprocess_for_model(image: Image.Image, target_size: int = 224) -> Image.Image:
    """Prepare an image for model inference or training by resizing and ensuring RGB."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return ImageOps.fit(image, (target_size, target_size), Image.LANCZOS)


def random_jpeg_compression(image: Image.Image, quality_range: Tuple[int, int] = (60, 90), probability: float = 0.8) -> Image.Image:
    """Randomly re-encode the image as JPEG to simulate compression artifacts."""
    if np.random.rand() > probability:
        return image
    quality = int(np.random.randint(*quality_range))
    return jpeg_compress(image, quality=quality)


def create_robust_variants(image: Image.Image, max_variants: int = 6) -> List[Image.Image]:
    """Create robust image variants to reduce false positives on low-quality real photos."""
    variants = [preprocess_for_model(image)]

    augmenters = [
        random_gaussian_blur,
        random_motion_blur,
        random_jpeg_compression,
        random_noise,
        random_brightness_contrast,
        random_color_jitter,
        random_crop_resize,
    ]

    for aug in augmenters:
        if len(variants) >= max_variants:
            break
        variant = aug(image)
        variant = preprocess_for_model(variant)
        if all(variant.tobytes() != existing.tobytes() for existing in variants):
            variants.append(variant)

    # Add one low-quality compression variant, if not present
    compressed = preprocess_for_model(jpeg_compress(image, quality=70))
    if all(compressed.tobytes() != existing.tobytes() for existing in variants):
        variants.append(compressed)

    return variants[:max_variants]


def resize_for_display(image: Image.Image, max_size: int = 800) -> Image.Image:
    """Resize image for display while keeping aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def get_image_stats(image: Image.Image) -> dict:
    """
    Extract basic image statistics that can hint at AI generation.
    AI images often have:
    - Very high sharpness (no natural blur/noise)
    - Unusual color distribution
    - Lack of natural noise patterns
    """
    img_array = np.array(image)

    # Basic stats
    stat = ImageStat.Stat(image)
    mean_brightness = float(np.mean(stat.mean))
    std_brightness = float(np.mean(stat.stddev))

    # Sharpness estimate using Laplacian variance
    gray = np.array(image.convert("L"), dtype=np.float32)
    laplacian = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0],
    ], dtype=np.float32)

    def convolve2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_h, a_w = a.shape
        b_h, b_w = b.shape
        out_h, out_w = a_h - b_h + 1, a_w - b_w + 1
        out = np.empty((out_h, out_w), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                out[i, j] = np.sum(a[i:i + b_h, j:j + b_w] * b)
        return out

    lap_img = convolve2d(gray, laplacian)
    sharpness = float(np.var(lap_img))

    # Color channel stats
    r_std = float(np.std(img_array[:, :, 0]))
    g_std = float(np.std(img_array[:, :, 1]))
    b_std = float(np.std(img_array[:, :, 2]))

    # Noise estimate — real photos have natural sensor noise
    blurred = np.array(image.filter(ImageFilter.GaussianBlur(radius=2)).convert("L"), dtype=np.float32)
    noise_level = float(np.std(gray - blurred))

    # Aspect ratio
    w, h = image.size
    aspect_ratio = round(w / h, 3)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "mean_brightness": round(mean_brightness, 2),
        "brightness_std": round(std_brightness, 2),
        "sharpness_score": round(sharpness, 2),
        "noise_level": round(noise_level, 4),
        "color_std_r": round(r_std, 2),
        "color_std_g": round(g_std, 2),
        "color_std_b": round(b_std, 2),
        "megapixels": round((w * h) / 1_000_000, 2),
    }


def compute_ela_score(image: Image.Image, quality: int = 90) -> Tuple[float, Image.Image]:
    """
    Error Level Analysis (ELA) — a forensic technique.
    Saves and recompresses image at given quality, then compares to original.
    AI generated images often show uniform ELA patterns.
    Real photos show varying ELA patterns from natural compression artifacts.

    Returns (ela_score, ela_visualization_image)
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")

    orig_array = np.array(image, dtype=np.float32)
    comp_array = np.array(compressed, dtype=np.float32)
    diff = np.abs(orig_array - comp_array)

    scale = 10
    ela_image = Image.fromarray(np.clip(diff * scale, 0, 255).astype(np.uint8))
    ela_score = float(np.std(diff))

    return round(ela_score, 4), ela_image


def get_frequency_analysis(image: Image.Image) -> dict:
    """
    Analyze frequency components using FFT.
    AI images often have unusual frequency patterns —
    they may lack the natural high-frequency noise of real cameras.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))

    h, w = magnitude.shape
    cx, cy = w // 2, h // 2

    low_freq_mask = np.zeros((h, w))
    radius = min(h, w) // 8
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    low_freq_mask[dist <= radius] = 1

    low_energy = float(np.mean(magnitude[low_freq_mask == 1]))
    high_energy = float(np.mean(magnitude[low_freq_mask == 0]))
    freq_ratio = round(low_energy / (high_energy + 1e-8), 4)

    return {
        "low_frequency_energy": round(low_energy, 4),
        "high_frequency_energy": round(high_energy, 4),
        "freq_ratio": freq_ratio,
    }
