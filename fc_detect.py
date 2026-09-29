"""
╔══════════════════════════════════════════════════════════════╗
║       🔍 FAKE vs REAL IMAGE DETECTOR  —  v6.0               ║
║                                                              ║
║  Methods (19 total):                                         ║
║  ── Forensic Signals ──────────────────────────────────────  ║
║   1. ELA (Error Level Analysis)  + visualization            ║
║   2. Noise / PRNU sensor pattern                            ║
║   3. EXIF metadata                                          ║
║   4. Compression artifact deep analysis                     ║
║  ── Physics & Optics ──────────────────────────────────────  ║
║   5. Color & lighting physics                               ║
║   6. Shadow & highlight physics                             ║
║   7. Geometric distortion (lens/perspective)                ║
║   8. Reflection & refraction physics                        ║
║   9. Background blur realism (bokeh)                        ║
║  10. Object boundary sharpness anomaly                      ║
║  ── Texture & Pattern ─────────────────────────────────────  ║
║  11. Texture analysis (Gabor + LBP + Wavelet)               ║
║  12. Frequency analysis (FFT + DCT)                         ║
║  ── Face Analysis ─────────────────────────────────────────  ║
║  13. Facial geometry & symmetry                             ║
║  14. Face landmark 3D consistency                           ║
║  ── Deep Learning ─────────────────────────────────────────  ║
║  15. CNN feature analysis (ResNet50)                        ║
║  16. HuggingFace AI detector                                ║
║  17. SDXL detector (Organika)                               ║
║  18. Deepfake/GAN detector (prithivMLmods)                  ║
║  19. CLIP zero-shot                                         ║
║                                                             ║
║  Output: Console report + ELA visualization image saved     ║
║  No custom training needed — pre-trained models only!       ║
╚══════════════════════════════════════════════════════════════╝

Install:  pip install torch torchvision transformers pillow numpy opencv-python scipy
Run:      Set image_file below → python fake_real_detector_v6.py
"""

import os
import io
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# APPROACH 1: ELA — Error Level Analysis
# ══════════════════════════════════════════════════════════════
def detect_with_ela(image_path: str, quality: int = 90, save_vis: bool = True) -> dict:
    """
    ELA — Error Level Analysis  (+visualization saved as PNG)

    HOW IT WORKS:
    ─────────────
    JPEG compression works in 8x8 pixel blocks. Every time you save
    a JPEG, each block loses a tiny bit of quality (lossy compression).

    Step 1: Re-save the image at a known quality (e.g. 90%)
    Step 2: Subtract re-saved image from original  →  ELA image
    Step 3: Analyze the difference (error levels)
    Step 4: Save amplified ELA image for visual inspection

    REAL photo logic:
      - Was originally saved by a camera at some quality level
      - Re-saving again causes MORE compression error
      - Result: ELA image shows HIGH, VARIED brightness across regions
      - Different surfaces (sky, face, cloth) have different error levels

    AI / Edited image logic:
      - AI images are generated fresh — no prior compression history
      - OR edited regions were re-compressed multiple times already
      - Re-saving causes LESS additional error (already compressed)
      - Result: ELA image shows LOW, UNIFORM brightness
      - Edited/pasted regions show different error level than surroundings

    DETECTION:
      - High mean ELA brightness  →  likely REAL (more compression delta)
      - Low std of ELA brightness →  likely FAKE (too uniform, no variation)
      - We combine both signals for final score

    VISUALIZATION:
      - Saved as <original_name>_ELA.png next to the input image
      - Bright regions = high compression error = likely original/real areas
      - Dark uniform regions = suspicious = possibly AI-generated/edited
    """
    try:
        print("[ELA] Running Error Level Analysis...")

        original = Image.open(image_path).convert("RGB")

        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        ela_image = ImageChops.difference(original, recompressed)

        scale = 10
        ela_array = np.array(ela_image).astype(np.float32) * scale
        ela_array = np.clip(ela_array, 0, 255)

        # ── Save ELA visualization ──
        vis_path = None
        if save_vis:
            try:
                ela_vis = Image.fromarray(ela_array.astype(np.uint8))
                base     = os.path.splitext(image_path)[0]
                vis_path = base + "_ELA_visualization.png"
                ela_vis.save(vis_path)
                print(f"[ELA] Visualization saved → {vis_path}")
            except Exception as ve:
                print(f"[ELA] Could not save visualization: {ve}")

        ela_mean = np.mean(ela_array)
        ela_std  = np.std(ela_array)
        ela_max  = np.max(ela_array)

        h, w = ela_array.shape[:2]
        quadrants = [
            ela_array[:h//2, :w//2],
            ela_array[:h//2, w//2:],
            ela_array[h//2:, :w//2],
            ela_array[h//2:, w//2:],
        ]
        quad_means     = [np.mean(q) for q in quadrants]
        quad_variation = np.std(quad_means)

        fake_score = 0
        reasons    = []

        if ela_mean < 8.0:
            fake_score += 30
            reasons.append(f"Low ELA mean={ela_mean:.2f} (too clean, minimal compression delta)")
        if ela_std < 12.0:
            fake_score += 30
            reasons.append(f"Low ELA std={ela_std:.2f} (uniform error, suspicious consistency)")
        if quad_variation < 3.0:
            fake_score += 20
            reasons.append(f"Low regional variation={quad_variation:.2f} (uniform across image)")
        if ela_max < 30.0:
            fake_score += 20
            reasons.append(f"Low peak error={ela_max:.2f} (no high-error edges/details)")

        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "ELA (Error Level Analysis)",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {
                "fake_probability": f"{fake_score:.0f}%",
                "real_probability": f"{real_score:.0f}%",
            },
            "ela_metrics": {
                "mean_brightness":    round(float(ela_mean), 3),
                "std_brightness":     round(float(ela_std), 3),
                "max_brightness":     round(float(ela_max), 3),
                "regional_variation": round(float(quad_variation), 3),
            },
            "ela_visualization": vis_path,
            "indicators": reasons if reasons else ["ELA patterns look natural (REAL)"],
            "success": True
        }

    except Exception as e:
        return {"method": "ELA (Error Level Analysis)", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 2: CNN-based Detection (ResNet50)
# ══════════════════════════════════════════════════════════════
def detect_with_cnn(image_path: str) -> dict:
    """
    CNN — Convolutional Neural Network Detection

    HOW IT WORKS:
    ─────────────
    We use ResNet50 pretrained on ImageNet as a feature extractor.
    CNNs learn hierarchical features:
      - Early layers: edges, textures, colors
      - Middle layers: parts, patterns
      - Deep layers: high-level semantics

    WHY CNN CAN DETECT FAKES (without retraining):
      Real images have:
        ✓ Natural texture gradients (rough→smooth transitions)
        ✓ Sensor noise patterns (slight grain from camera CCD/CMOS)
        ✓ Natural depth of field blur
        ✓ Realistic lighting gradients with imperfections

      AI images have:
        ✗ Overly smooth textures — CNN detects unusually low texture entropy
        ✗ Missing sensor noise — very low activation in noise-sensitive filters
        ✗ Symmetric/repetitive patterns — CNN finds unnaturally regular features
        ✗ Spectral artifacts in deep feature maps

    We extract features from multiple CNN layers and analyze:
      1. Texture complexity     (early layer activations)
      2. Feature activation entropy  (mid layer)
      3. Pattern regularity     (deep layer)
      4. Activation sparsity    (overall)

    These 4 signals are combined into a fake probability score.
    """
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms

        print("[CNN] Loading ResNet50 for feature extraction...")

        # ── Load pretrained ResNet50 ──
        resnet = models.resnet50(weights='DEFAULT')
        resnet.eval()

        # ── Hook storage for multi-layer features ──
        layer_features = {}

        def make_hook(name):
            def hook(module, input, output):
                layer_features[name] = output.detach()
            return hook

        # Register hooks at 3 different depths:
        # layer1 = early (texture/edges)
        # layer3 = mid (patterns/parts)
        # layer4 = deep (semantic)
        resnet.layer1.register_forward_hook(make_hook("early"))
        resnet.layer3.register_forward_hook(make_hook("mid"))
        resnet.layer4.register_forward_hook(make_hook("deep"))

        # ── Preprocess image ──
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

        # ── Forward pass ──
        with torch.no_grad():
            _ = resnet(tensor)

        # ── Analysis per layer ──
        fake_score = 0
        reasons = []
        layer_stats = {}

        for layer_name, feat_tensor in layer_features.items():
            feat = feat_tensor.numpy()  # Shape: (1, C, H, W)

            # Mean activation per channel
            channel_means = feat.mean(axis=(0, 2, 3))  # (C,)
            # Std of activations
            activation_std = float(feat.std())
            # Sparsity: fraction of near-zero activations (ReLU zeros)
            sparsity = float(np.mean(np.abs(feat) < 0.01))
            # Entropy of activation distribution
            hist, _ = np.histogram(feat.flatten(), bins=50, density=True)
            hist = hist + 1e-10
            entropy = float(-np.sum(hist * np.log(hist)))
            # Channel variation: how different are channels from each other?
            channel_variation = float(np.std(channel_means))

            layer_stats[layer_name] = {
                "activation_std":    round(activation_std, 4),
                "sparsity":          round(sparsity, 4),
                "entropy":           round(entropy, 4),
                "channel_variation": round(channel_variation, 4),
            }

            # ── Scoring per layer ──
            # Early layer: Real images have rich texture → high std, high entropy
            if layer_name == "early":
                if activation_std < 0.3:
                    fake_score += 10
                    reasons.append("Early CNN: low texture std (too smooth)")
                if entropy < 3.5:
                    fake_score += 10
                    reasons.append("Early CNN: low texture entropy (uniform surface)")

            # Mid layer: Real images have varied pattern activations
            elif layer_name == "mid":
                if channel_variation < 0.1:
                    fake_score += 15
                    reasons.append("Mid CNN: low channel variation (repetitive patterns)")
                if sparsity > 0.85:
                    fake_score += 15
                    reasons.append("Mid CNN: high sparsity (few active features)")

            # Deep layer: Real images activate diverse semantic features
            elif layer_name == "deep":
                if activation_std < 0.5:
                    fake_score += 15
                    reasons.append("Deep CNN: low semantic activation variance")
                if channel_variation < 0.2:
                    fake_score += 15
                    reasons.append("Deep CNN: uniform semantic channels (unnatural)")

        # Cap at 100
        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "CNN Feature Analysis (ResNet50)",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {
                "fake_probability": f"{fake_score:.0f}%",
                "real_probability": f"{real_score:.0f}%",
            },
            "layer_stats": layer_stats,
            "indicators": reasons if reasons else ["CNN activation patterns look natural (REAL)"],
            "success": True
        }

    except Exception as e:
        return {"method": "CNN Feature Analysis (ResNet50)", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 3: HuggingFace Pre-trained AI Detector
# ══════════════════════════════════════════════════════════════
def detect_with_hf_model(image_path: str) -> dict:
    """
    Uses 'umm-maybe/AI-image-detector' — specifically trained to detect
    AI-generated images (DALL-E, Midjourney, Stable Diffusion, etc.)
    """
    try:
        from transformers import pipeline
        print("[HF Model] Loading AI image detector...")
        detector = pipeline("image-classification", model="umm-maybe/AI-image-detector")
        result = detector(image_path)

        label_map = {}
        for item in result:
            label_map[item['label'].lower()] = round(item['score'] * 100, 2)

        artificial = label_map.get('artificial', label_map.get('ai', 0))
        human      = label_map.get('human',      label_map.get('real', 0))

        prediction = "FAKE (AI-Generated)" if artificial > human else "REAL"
        confidence = max(artificial, human)

        return {
            "method": "HuggingFace AI Detector",
            "prediction": prediction,
            "confidence": f"{confidence:.1f}%",
            "scores": label_map,
            "success": True
        }
    except Exception as e:
        return {"method": "HuggingFace AI Detector", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 4: CLIP Zero-Shot
# ══════════════════════════════════════════════════════════════
def detect_with_clip(image_path: str) -> dict:
    """
    Uses OpenAI CLIP for zero-shot classification via text prompts.
    No fine-tuning needed.
    """
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch

        print("[CLIP] Loading CLIP model...")
        model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        image = Image.open(image_path).convert("RGB")

        text_prompts = [
            "a real photograph taken by a camera",
            "an AI-generated synthetic image",
            "a photo of a real person or scene",
            "an artificial image created by AI like DALL-E or Midjourney",
        ]

        inputs = processor(text=text_prompts, images=image,
                           return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).numpy()[0]

        real_score = (probs[0] + probs[2]) / 2 * 100
        fake_score = (probs[1] + probs[3]) / 2 * 100

        prediction = "FAKE (AI-Generated)" if fake_score > real_score else "REAL"

        return {
            "method": "CLIP Zero-Shot",
            "prediction": prediction,
            "confidence": f"{max(real_score, fake_score):.1f}%",
            "scores": {
                "real":         f"{real_score:.1f}%",
                "ai_generated": f"{fake_score:.1f}%",
            },
            "success": True
        }
    except Exception as e:
        return {"method": "CLIP Zero-Shot", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 5: Frequency / Statistical Analysis
# ══════════════════════════════════════════════════════════════
def detect_with_frequency_analysis(image_path: str) -> dict:
    """
    Classical ML — no deep learning.
    Analyzes FFT frequency patterns, noise levels, DCT uniformity.
    AI images often lack natural noise & have suspicious frequency artifacts.
    """
    try:
        import cv2

        print("[Frequency] Running FFT/DCT frequency analysis...")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image with OpenCV")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # FFT analysis
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)
        h, w = magnitude.shape
        center_energy = np.mean(magnitude[h//4:3*h//4, w//4:3*w//4])
        edge_energy   = np.mean(magnitude)
        freq_ratio    = center_energy / (edge_energy + 1e-8)

        # Noise level
        laplacian_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()

        # Color saturation uniformity
        hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_std = np.std(hsv[:, :, 1])

        # DCT block uniformity
        dct_stds = []
        step = 8
        for i in range(0, gray.shape[0] - step, step):
            for j in range(0, gray.shape[1] - step, step):
                block = gray[i:i+step, j:j+step]
                dct_stds.append(np.std(cv2.dct(block)))
        dct_uniformity = np.std(dct_stds)

        fake_indicators = 0
        reasons = []

        if freq_ratio < 1.8:
            fake_indicators += 1
            reasons.append("Suspicious FFT frequency distribution")
        if laplacian_var < 30:
            fake_indicators += 1
            reasons.append(f"Unusually low noise (Laplacian={laplacian_var:.1f})")
        if sat_std < 25:
            fake_indicators += 1
            reasons.append("Overly uniform color saturation")
        if dct_uniformity < 15:
            fake_indicators += 1
            reasons.append("Low DCT coefficient variance")

        fake_prob = (fake_indicators / 4) * 100
        real_prob = 100 - fake_prob
        prediction = "FAKE (AI-Generated)" if fake_prob >= 50 else "REAL"

        return {
            "method": "Frequency/Statistical Analysis (FFT+DCT)",
            "prediction": prediction,
            "confidence": f"{max(fake_prob, real_prob):.0f}%",
            "scores": {
                "fake_probability": f"{fake_prob:.0f}%",
                "real_probability": f"{real_prob:.0f}%",
            },
            "indicators": reasons if reasons else ["No suspicious frequency patterns"],
            "raw_metrics": {
                "freq_ratio":    round(freq_ratio, 3),
                "noise_level":   round(float(laplacian_var), 2),
                "sat_uniformity": round(float(sat_std), 2),
                "dct_uniformity": round(float(dct_uniformity), 2),
            },
            "success": True
        }
    except Exception as e:
        return {"method": "Frequency/Statistical Analysis (FFT+DCT)",
                "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 6: Organika SDXL Detector
# ══════════════════════════════════════════════════════════════
def detect_with_sdxl_detector(image_path: str) -> dict:
    """
    Organika/sdxl-detector — Forensics-grade HuggingFace Model

    HOW IT WORKS:
    ─────────────
    This model is fine-tuned specifically to detect images generated by
    Stable Diffusion XL (SDXL), one of the most popular AI image generators.

    It was trained on:
      - Real photos from LAION dataset
      - AI-generated images from SDXL, SD 1.5, SD 2.x pipelines

    Architecture: Vision Transformer (ViT) fine-tuned as binary classifier
      → Input: any image
      → Output: "artificial" (AI-generated) or "real" probability

    WHY IT'S BETTER FOR SDXL/SD IMAGES:
      The general HuggingFace detector (umm-maybe) is broad.
      Organika is specifically tuned for Stable Diffusion artifacts:
        - SD's characteristic over-smooth skin texture
        - SDXL's specific noise patterns in backgrounds
        - Characteristic color distribution of SD outputs
        - Specific artifacts in hair, hands, and fine details

    This is especially powerful when combined with umm-maybe — together
    they cover a wider range of AI generators.
    """
    try:
        from transformers import pipeline
        print("[SDXL Detector] Loading Organika SDXL detector...")

        detector = pipeline(
            "image-classification",
            model="Organika/sdxl-detector"
        )
        result = detector(image_path)

        label_map = {}
        for item in result:
            label_map[item['label'].lower()] = round(item['score'] * 100, 2)

        # Labels: 'artificial' vs 'real' (or similar)
        artificial = label_map.get('artificial', label_map.get('fake', label_map.get('ai', 0)))
        real       = label_map.get('real',       label_map.get('human', 0))

        # If neither found, check which label has highest score
        if artificial == 0 and real == 0:
            sorted_labels = sorted(label_map.items(), key=lambda x: x[1], reverse=True)
            top_label = sorted_labels[0][0] if sorted_labels else "unknown"
            artificial = label_map.get(top_label, 0) if 'art' in top_label or 'fake' in top_label or 'ai' in top_label else 0
            real = 100 - artificial

        prediction = "FAKE (AI-Generated)" if artificial > real else "REAL"
        confidence = max(artificial, real)

        return {
            "method": "SDXL Detector (Organika)",
            "prediction": prediction,
            "confidence": f"{confidence:.1f}%",
            "scores": label_map,
            "success": True
        }
    except Exception as e:
        return {"method": "SDXL Detector (Organika)", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 7: Deepfake / GAN Image Detector
# ══════════════════════════════════════════════════════════════
def detect_with_deepfake_detector(image_path: str) -> dict:
    """
    prithivMLmods/Deep-Fake-Detector-Model — GAN & Deepfake specialist

    HOW IT WORKS:
    ─────────────
    This model focuses on GAN-generated images and deepfakes — a different
    class of AI fakes compared to diffusion models (SD, DALL-E).

    GANs (Generative Adversarial Networks) produce:
      - DeepFakes (swapped faces)
      - StyleGAN portraits ("This Person Does Not Exist")
      - GAN-synthesized scenes

    Diffusion models (SD, DALL-E, Midjourney) produce:
      - Text-to-image generations
      - Image editing / inpainting

    WHY BOTH DETECTORS ARE NEEDED:
      ┌─────────────────┬──────────────────────────────────────┐
      │ umm-maybe       │ Good for diffusion models broadly    │
      │ Organika        │ Best for Stable Diffusion / SDXL     │
      │ DeepFake Det.   │ Best for GANs, StyleGAN, Deepfakes   │
      └─────────────────┴──────────────────────────────────────┘
      Together these 3 cover the full spectrum of AI-generated images.

    Architecture: EfficientNet / ViT fine-tuned on GAN datasets
      Trained on: FaceForensics++, DFDC, StyleGAN2, ProGAN datasets

    GAN-specific artifacts it detects:
      - Checkerboard artifacts from transposed convolutions
      - Inconsistent eye reflections (GAN faces)
      - Missing / wrong details in backgrounds
      - Characteristic GAN frequency spectrum peaks
    """
    try:
        from transformers import pipeline
        print("[Deepfake Detector] Loading GAN/Deepfake detector...")

        detector = pipeline(
            "image-classification",
            model="prithivMLmods/Deep-Fake-Detector-Model"
        )
        result = detector(image_path)

        label_map = {}
        for item in result:
            label_map[item['label'].lower()] = round(item['score'] * 100, 2)

        # Try common label names
        fake_score = 0
        real_score = 0
        for label, score in label_map.items():
            if any(k in label for k in ['fake', 'artificial', 'generated', 'deepfake', 'synthetic']):
                fake_score += score
            elif any(k in label for k in ['real', 'human', 'authentic', 'genuine']):
                real_score += score

        # Fallback: if labels not recognized, use top label
        if fake_score == 0 and real_score == 0:
            sorted_items = sorted(label_map.items(), key=lambda x: x[1], reverse=True)
            if sorted_items:
                top_label, top_score = sorted_items[0]
                # Assume first label is "fake" if it has >50% and we can't parse it
                fake_score = top_score if top_score > 50 else 0
                real_score = 100 - fake_score

        prediction = "FAKE (AI-Generated)" if fake_score > real_score else "REAL"
        confidence = max(fake_score, real_score)

        return {
            "method": "Deepfake/GAN Detector (prithivMLmods)",
            "prediction": prediction,
            "confidence": f"{confidence:.1f}%",
            "scores": label_map,
            "success": True
        }
    except Exception as e:
        return {"method": "Deepfake/GAN Detector (prithivMLmods)", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 8: EXIF Metadata Analysis
# ══════════════════════════════════════════════════════════════
def detect_with_exif(image_path: str) -> dict:
    """
    EXIF Metadata Analysis — Camera fingerprint detection

    HOW IT WORKS:
    ─────────────
    Real camera photos embed EXIF metadata automatically:
      - Camera make/model (Canon EOS R5, iPhone 15 Pro, etc.)
      - Lens focal length, aperture, shutter speed, ISO
      - GPS coordinates (if enabled)
      - Date/time of capture
      - Color space, white balance, metering mode
      - Thumbnail of the image

    AI-generated images:
      - Have NO EXIF data (generated from noise, not captured)
      - OR have suspicious/minimal EXIF (added by tool, not camera)
      - Missing camera-specific fields like ExposureTime, FNumber
      - No GPS data, no lens info, no flash info

    DETECTION LOGIC:
      We score based on presence/absence of camera-specific EXIF fields.
      The more camera fields present → more likely REAL.
      Missing all camera fields → strong FAKE indicator.

    NOTE: EXIF can be stripped (privacy tools, social media).
    So missing EXIF alone is not conclusive — it's one signal among many.
    """
    try:
        from PIL.ExifTags import TAGS
        print("[EXIF] Analyzing image metadata...")

        image = Image.open(image_path)
        exif_data = image._getexif() if hasattr(image, '_getexif') else None

        fake_score = 0
        reasons = []
        exif_info = {}

        if exif_data is None or len(exif_data) == 0:
            # No EXIF at all — strong fake indicator
            fake_score = 70
            reasons.append("No EXIF metadata found (AI images have no camera data)")
            exif_info["status"] = "NO EXIF"
        else:
            # Parse EXIF tags
            parsed = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                try:
                    parsed[tag_name] = str(value)[:80]  # truncate long values
                except:
                    pass

            exif_info["tags_found"] = len(parsed)

            # ── Camera-specific fields check ──
            camera_fields = {
                "Make":          "Camera manufacturer",
                "Model":         "Camera model",
                "ExposureTime":  "Shutter speed",
                "FNumber":       "Aperture f/stop",
                "ISOSpeedRatings": "ISO sensitivity",
                "FocalLength":   "Lens focal length",
                "Flash":         "Flash info",
                "WhiteBalance":  "White balance setting",
                "MeteringMode":  "Metering mode",
                "LensModel":     "Lens model",
            }

            found_camera_fields = []
            missing_camera_fields = []

            for field, description in camera_fields.items():
                if field in parsed:
                    found_camera_fields.append(f"{field}={parsed[field][:30]}")
                    exif_info[field] = parsed[field][:50]
                else:
                    missing_camera_fields.append(field)

            # Score based on missing camera fields
            missing_ratio = len(missing_camera_fields) / len(camera_fields)

            if missing_ratio >= 0.9:
                fake_score += 60
                reasons.append(f"Almost no camera EXIF fields ({len(found_camera_fields)}/{len(camera_fields)} found)")
            elif missing_ratio >= 0.7:
                fake_score += 35
                reasons.append(f"Few camera EXIF fields ({len(found_camera_fields)}/{len(camera_fields)} found)")
            elif missing_ratio >= 0.5:
                fake_score += 15
                reasons.append(f"Some camera EXIF fields missing ({len(found_camera_fields)}/{len(camera_fields)} found)")

            # Software field: AI tools sometimes add this
            if "Software" in parsed:
                sw = parsed["Software"].lower()
                ai_tools = ["stable diffusion", "midjourney", "dall-e", "runway",
                            "adobe firefly", "generative", "ai", "diffusion"]
                if any(tool in sw for tool in ai_tools):
                    fake_score += 30
                    reasons.append(f"AI software detected in EXIF: {parsed['Software'][:40]}")
                exif_info["Software"] = parsed["Software"][:50]

            if found_camera_fields:
                exif_info["camera_fields"] = found_camera_fields[:5]

            if not reasons:
                reasons.append(f"Camera EXIF present: {len(found_camera_fields)}/{len(camera_fields)} fields found")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "EXIF Metadata Analysis",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {
                "fake_probability": f"{fake_score:.0f}%",
                "real_probability": f"{real_score:.0f}%",
            },
            "exif_info": exif_info,
            "indicators": reasons,
            "success": True
        }
    except Exception as e:
        return {"method": "EXIF Metadata Analysis", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 9: Noise Pattern / Camera Sensor Analysis (PRNU)
# ══════════════════════════════════════════════════════════════
def detect_with_noise_analysis(image_path: str) -> dict:
    """
    Noise Pattern Analysis — Camera sensor fingerprint (PRNU)

    HOW IT WORKS:
    ─────────────
    Every digital camera sensor has a unique noise pattern called
    PRNU (Photo Response Non-Uniformity) — like a fingerprint.

    Real photos:
      ✓ Have natural sensor noise (Gaussian + Poisson distribution)
      ✓ Noise is spatially random but statistically consistent
      ✓ Different channels (R,G,B) have correlated noise patterns
      ✓ Noise increases in darker regions (natural physics)

    AI-generated images:
      ✗ No sensor noise — mathematically generated pixels
      ✗ Or have fake/patterned noise added by the model
      ✗ Noise distribution is NOT Gaussian — too uniform or structured
      ✗ Channel noise independence (R,G,B noise not correlated)
      ✗ Noise does NOT increase in dark regions

    DETECTION SIGNALS:
      1. Noise extraction: image - gaussian_blur(image) = noise layer
      2. Noise distribution analysis (should be ~Gaussian for real)
      3. Spatial autocorrelation (real noise = random, AI = structured)
      4. Channel noise correlation (R-G-B should be correlated in real)
      5. Local noise variance vs brightness (should increase in dark areas)
    """
    try:
        import cv2
        from scipy import stats as scipy_stats

        print("[Noise] Analyzing camera sensor noise patterns...")

        img = cv2.imread(image_path).astype(np.float32)
        if img is None:
            raise ValueError("Cannot read image")

        fake_score = 0
        reasons = []
        noise_stats = {}

        # ── Step 1: Extract noise layer ──
        # Noise = original - smoothed version (high-frequency residual)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        noise = img - blurred  # Shape: (H, W, 3)

        # ── Step 2: Noise distribution analysis ──
        # Real camera noise ≈ Gaussian distribution
        noise_flat = noise.flatten()
        noise_mean = float(np.mean(noise_flat))
        noise_std  = float(np.std(noise_flat))

        # Kurtosis: Gaussian = 3.0. AI noise often deviates significantly
        noise_kurtosis = float(np.mean((noise_flat - noise_mean)**4) / (noise_std**4 + 1e-8))

        noise_stats["mean"]     = round(noise_mean, 4)
        noise_stats["std"]      = round(noise_std, 4)
        noise_stats["kurtosis"] = round(noise_kurtosis, 3)

        # Real images: kurtosis close to 3 (Gaussian)
        # AI images: kurtosis either too high (spiky) or too low (flat)
        if abs(noise_kurtosis - 3.0) > 5.0:
            fake_score += 20
            reasons.append(f"Non-Gaussian noise distribution (kurtosis={noise_kurtosis:.2f}, expected ~3.0)")

        # ── Step 3: Spatial autocorrelation of noise ──
        # Real noise: low autocorrelation (random)
        # AI noise: higher autocorrelation (structured/patterned)
        noise_gray = noise.mean(axis=2)  # (H, W)
        h, w = noise_gray.shape
        # Compare adjacent pixels' noise values
        autocorr_h = float(np.corrcoef(noise_gray[:h-1, :].flatten(),
                                        noise_gray[1:, :].flatten())[0, 1])
        autocorr_w = float(np.corrcoef(noise_gray[:, :w-1].flatten(),
                                        noise_gray[:, 1:].flatten())[0, 1])
        avg_autocorr = abs((autocorr_h + autocorr_w) / 2)
        noise_stats["spatial_autocorr"] = round(avg_autocorr, 4)

        if avg_autocorr > 0.3:
            fake_score += 20
            reasons.append(f"Structured noise pattern (autocorr={avg_autocorr:.3f}, real noise should be < 0.3)")

        # ── Step 4: Channel noise correlation ──
        # Real camera: R,G,B channels have correlated noise (same sensor)
        # AI image: channels may have independent/uncorrelated noise
        noise_r = noise[:, :, 2].flatten()
        noise_g = noise[:, :, 1].flatten()
        noise_b = noise[:, :, 0].flatten()

        corr_rg = float(np.corrcoef(noise_r, noise_g)[0, 1])
        corr_rb = float(np.corrcoef(noise_r, noise_b)[0, 1])
        avg_channel_corr = (abs(corr_rg) + abs(corr_rb)) / 2
        noise_stats["channel_correlation"] = round(avg_channel_corr, 4)

        if avg_channel_corr < 0.3:
            fake_score += 20
            reasons.append(f"Low channel noise correlation (corr={avg_channel_corr:.3f}, real cameras show > 0.3)")

        # ── Step 5: Noise vs brightness relationship ──
        # Real cameras: noise increases in dark regions (shot noise physics)
        # AI images: uniform noise regardless of brightness
        gray_img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        noise_mag = np.abs(noise_gray)

        # Divide into brightness bins and check noise level per bin
        bright_mask = gray_img > 128
        dark_mask   = gray_img <= 128

        noise_in_bright = float(np.mean(noise_mag[bright_mask])) if bright_mask.any() else 0
        noise_in_dark   = float(np.mean(noise_mag[dark_mask]))   if dark_mask.any()   else 0
        noise_stats["noise_bright"] = round(noise_in_bright, 4)
        noise_stats["noise_dark"]   = round(noise_in_dark, 4)

        # Real: dark regions should have more noise than bright (shot noise)
        if noise_in_dark < noise_in_bright * 0.8:
            fake_score += 20
            reasons.append(f"Noise not higher in dark regions (physics violation: dark={noise_in_dark:.3f}, bright={noise_in_bright:.3f})")

        # ── Step 6: Overall noise level ──
        if noise_std < 1.5:
            fake_score += 20
            reasons.append(f"Extremely low noise level (std={noise_std:.3f}) — AI images are too clean")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Noise/Sensor Pattern Analysis (PRNU)",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {
                "fake_probability": f"{fake_score:.0f}%",
                "real_probability": f"{real_score:.0f}%",
            },
            "noise_stats": noise_stats,
            "indicators": reasons if reasons else ["Noise patterns consistent with real camera sensor"],
            "success": True
        }
    except ImportError:
        # scipy not available — use simplified version
        try:
            import cv2
            print("[Noise] Running simplified noise analysis (no scipy)...")
            img = cv2.imread(image_path).astype(np.float32)
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            noise = img - blurred
            noise_std = float(np.std(noise))
            fake_score = 60 if noise_std < 1.5 else 0
            prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"
            return {
                "method": "Noise/Sensor Pattern Analysis (PRNU)",
                "prediction": prediction,
                "confidence": f"{max(fake_score, 100-fake_score):.0f}%",
                "scores": {"fake_probability": f"{fake_score:.0f}%", "real_probability": f"{100-fake_score:.0f}%"},
                "indicators": [f"Noise std={noise_std:.3f}"],
                "success": True
            }
        except Exception as e2:
            return {"method": "Noise/Sensor Pattern Analysis (PRNU)", "success": False, "error": str(e2)}
    except Exception as e:
        return {"method": "Noise/Sensor Pattern Analysis (PRNU)", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 10: Color & Lighting Physics Check
# ══════════════════════════════════════════════════════════════
def detect_with_color_physics(image_path: str) -> dict:
    """
    Color & Lighting Physics Analysis

    HOW IT WORKS:
    ─────────────
    Real-world lighting follows physical laws that AI models often violate:

    1. COLOR TEMPERATURE CONSISTENCY
       Real photos: light source has one color temperature
         Sunlight = 5500K (neutral/slightly blue)
         Indoor tungsten = 2700K (warm/orange)
         → Highlights and shadows are consistent with ONE light source
       AI images: often mix color temperatures unnaturally

    2. SHADOW-HIGHLIGHT RELATIONSHIP
       Real physics: bright areas → warm highlights, cool shadows (or vice versa)
       AI images: shadows and highlights may have wrong color relationship

    3. COLOR CHANNEL HISTOGRAM SHAPE
       Real photos have characteristic histogram shapes:
         - Natural rolloff at extremes (few pure black/white pixels)
         - Smooth distribution (no sudden spikes)
       AI images: often have:
         - Clipped channels (too many pure 0 or 255 values)
         - Unnaturally smooth/perfect histograms
         - Suspicious peaks at specific values

    4. CHROMATIC ABERRATION
       Real lenses: slight color fringing at high-contrast edges
       AI images: perfect edges with no chromatic aberration

    5. COLOR GAMUT ANALYSIS
       Real photos: colors clustered in natural skin/sky/foliage zones
       AI images: often have oversaturated or impossible color combinations
    """
    try:
        import cv2
        print("[Color Physics] Analyzing color & lighting physics...")

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError("Cannot read image")

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        img_lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        fake_score = 0
        reasons = []
        color_stats = {}

        r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

        # ── Check 1: Histogram clipping ──
        # Real photos rarely have pixels at exactly 0 or 255
        clip_low  = float(np.mean((img_rgb < 2)))    # fraction near pure black
        clip_high = float(np.mean((img_rgb > 253)))  # fraction near pure white
        color_stats["clip_low"]  = round(clip_low, 4)
        color_stats["clip_high"] = round(clip_high, 4)

        if clip_high > 0.05:
            fake_score += 15
            reasons.append(f"Overexposed highlights: {clip_high*100:.1f}% pixels at max brightness (AI over-sharpening)")

        # ── Check 2: Color channel balance ──
        # Real photos have natural color temperature: channels are NOT perfectly equal
        r_mean = float(np.mean(r))
        g_mean = float(np.mean(g))
        b_mean = float(np.mean(b))
        channel_balance = np.std([r_mean, g_mean, b_mean])
        color_stats["r_mean"] = round(r_mean, 2)
        color_stats["g_mean"] = round(g_mean, 2)
        color_stats["b_mean"] = round(b_mean, 2)
        color_stats["channel_balance"] = round(channel_balance, 3)

        # Too equal = AI (no natural color temperature tint)
        if channel_balance < 3.0:
            fake_score += 15
            reasons.append(f"Unnaturally balanced color channels (std={channel_balance:.2f}) — real photos have color temperature tint")

        # ── Check 3: Saturation distribution ──
        sat = img_hsv[:,:,1]
        sat_mean = float(np.mean(sat))
        sat_std  = float(np.std(sat))
        # Fraction of oversaturated pixels (sat > 220 in 0-255 range)
        oversaturated = float(np.mean(sat > 220))
        color_stats["saturation_mean"] = round(sat_mean, 2)
        color_stats["saturation_std"]  = round(sat_std, 2)
        color_stats["oversaturated_pct"] = round(oversaturated * 100, 2)

        if oversaturated > 0.15:
            fake_score += 15
            reasons.append(f"Oversaturated colors: {oversaturated*100:.1f}% pixels (AI often over-saturates)")
        if sat_std < 20:
            fake_score += 10
            reasons.append(f"Uniform saturation (std={sat_std:.1f}) — unnatural, real scenes have varied saturation")

        # ── Check 4: Chromatic aberration check ──
        # Real lenses: R and B channels are slightly misaligned at edges
        # Check correlation between R and B channels at high-gradient areas
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = edges > 0

        if edge_mask.sum() > 100:
            r_at_edges = r[edge_mask]
            b_at_edges = b[edge_mask]
            rb_corr_at_edges = float(np.corrcoef(r_at_edges, b_at_edges)[0, 1])
            color_stats["rb_edge_correlation"] = round(rb_corr_at_edges, 4)

            # Real: R-B slightly decorrelated at edges (chromatic aberration)
            # AI: R-B perfectly correlated (no physical lens distortion)
            if rb_corr_at_edges > 0.97:
                fake_score += 15
                reasons.append(f"No chromatic aberration detected (R-B corr={rb_corr_at_edges:.3f}) — real lenses show slight color fringing")

        # ── Check 5: LAB color space naturalness ──
        # In LAB space, real photos cluster in natural color zones
        a_channel = img_lab[:,:,1]  # green-red axis
        b_channel = img_lab[:,:,2]  # blue-yellow axis
        # Real photos: a,b channels have moderate spread
        a_std = float(np.std(a_channel))
        b_std = float(np.std(b_channel))
        color_stats["lab_a_std"] = round(a_std, 2)
        color_stats["lab_b_std"] = round(b_std, 2)

        if a_std > 35 or b_std > 35:
            fake_score += 15
            reasons.append(f"Extreme color spread in LAB space (a_std={a_std:.1f}, b_std={b_std:.1f}) — unnatural color palette")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Color & Lighting Physics",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {
                "fake_probability": f"{fake_score:.0f}%",
                "real_probability": f"{real_score:.0f}%",
            },
            "color_stats": color_stats,
            "indicators": reasons if reasons else ["Color & lighting physics look natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Color & Lighting Physics", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 11: Facial Geometry & Symmetry Analysis
# ══════════════════════════════════════════════════════════════
def detect_with_facial_geometry(image_path: str) -> dict:
    """
    Facial Geometry & Symmetry Analysis

    HOW IT WORKS:
    ─────────────
    AI face generators (GANs, diffusion models) produce faces with specific
    geometric artifacts that differ from real human faces:

    1. FACE DETECTION
       Uses OpenCV's pre-trained Haar cascade / DNN face detector.
       If no face found → skip this method (not applicable).

    2. SYMMETRY ANALYSIS
       Real faces: slightly asymmetric (humans are naturally asymmetric)
       AI faces: often TOO symmetric (generators favor symmetric outputs)
       Method: Mirror left half, compare with right half → symmetry score

    3. EYE ALIGNMENT
       Real faces: eyes at a natural angle, slight variation
       AI faces: eyes often unnaturally level/horizontal
       Method: Detect eye regions, measure alignment angle

    4. SKIN TEXTURE SMOOTHNESS
       Real faces: pores, fine lines, subtle texture variation
       AI faces: skin is unnaturally smooth ("porcelain skin" effect)
       Method: Local variance in skin region

    5. FACIAL PROPORTION CHECK
       Real faces follow golden ratio / natural proportions
       AI faces sometimes have slight proportion errors (too-large eyes,
       unusual nose-to-mouth distances, etc.)
    """
    try:
        import cv2
        print("[Facial] Analyzing facial geometry & symmetry...")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ── Step 1: Face Detection ──
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            return {
                "method": "Facial Geometry & Symmetry",
                "prediction": "REAL",   # No face → can't judge, default neutral
                "confidence": "50%",
                "scores": {"fake_probability": "50%", "real_probability": "50%"},
                "indicators": ["No face detected — facial analysis skipped (neutral score)"],
                "success": True
            }

        fake_score = 0
        reasons = []
        geo_stats = {}
        geo_stats["faces_detected"] = int(len(faces))

        # Analyze the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi     = img[y:y+h, x:x+w]
        face_gray    = gray[y:y+h, x:x+w]

        # ── Step 2: Facial Symmetry Analysis ──
        # Mirror the left half and compare with right half
        fh, fw = face_gray.shape
        left_half  = face_gray[:, :fw//2]
        right_half = face_gray[:, fw//2:]
        right_flipped = cv2.flip(right_half, 1)

        # Resize to same width if needed
        min_w = min(left_half.shape[1], right_flipped.shape[1])
        left_half     = left_half[:, :min_w]
        right_flipped = right_flipped[:, :min_w]

        # Symmetry score: 1 = perfect mirror, 0 = completely different
        diff = np.abs(left_half.astype(np.float32) - right_flipped.astype(np.float32))
        symmetry_score = 1.0 - (float(np.mean(diff)) / 128.0)
        symmetry_score = max(0.0, min(1.0, symmetry_score))
        geo_stats["symmetry_score"] = round(symmetry_score, 4)

        # Real faces: symmetry ~0.75–0.88 (slightly asymmetric)
        # AI faces: symmetry often > 0.90 (too perfect)
        if symmetry_score > 0.92:
            fake_score += 30
            reasons.append(f"Unnaturally high facial symmetry ({symmetry_score:.3f}) — real faces are slightly asymmetric")
        elif symmetry_score > 0.88:
            fake_score += 15
            reasons.append(f"Suspiciously high symmetry ({symmetry_score:.3f})")

        # ── Step 3: Eye Detection & Alignment ──
        eyes = eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
        )

        if len(eyes) >= 2:
            # Sort eyes by x position (left, right)
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            eye1 = eyes_sorted[0]
            eye2 = eyes_sorted[1]

            # Centers of each eye
            cx1 = eye1[0] + eye1[2]//2
            cy1 = eye1[1] + eye1[3]//2
            cx2 = eye2[0] + eye2[2]//2
            cy2 = eye2[1] + eye2[3]//2

            # Eye alignment angle (real faces: slight variation, AI: too level)
            import math
            eye_angle = abs(math.degrees(math.atan2(cy2 - cy1, cx2 - cx1)))
            geo_stats["eye_angle_deg"] = round(eye_angle, 2)

            # Real faces have a slight tilt; perfectly horizontal eyes are suspicious
            if eye_angle < 1.0:
                fake_score += 20
                reasons.append(f"Eyes are perfectly horizontal (angle={eye_angle:.2f}°) — unnaturally level")

            # Eye size ratio (should be close to 1.0 for real faces)
            eye1_area = eye1[2] * eye1[3]
            eye2_area = eye2[2] * eye2[3]
            eye_size_ratio = min(eye1_area, eye2_area) / (max(eye1_area, eye2_area) + 1e-8)
            geo_stats["eye_size_ratio"] = round(eye_size_ratio, 3)

            # FIX: eye_ratio=0.29 is a clear GAN artifact — stronger penalty
            if eye_size_ratio < 0.35:
                fake_score += 35
                reasons.append(f"Eyes have drastically different sizes (ratio={eye_size_ratio:.2f}) — strong GAN artifact")
            elif eye_size_ratio < 0.5:
                fake_score += 20
                reasons.append(f"Eyes have very different sizes (ratio={eye_size_ratio:.2f}) — GAN artifact")
            elif eye_size_ratio < 0.65:
                fake_score += 10
                reasons.append(f"Slight eye size asymmetry (ratio={eye_size_ratio:.2f})")

        # ── Step 4: Skin Texture Smoothness ──
        # AI faces have unnaturally smooth skin
        # Measure local variance in the face region
        local_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
        geo_stats["skin_texture_variance"] = round(local_var, 2)

        # FIX: skin_var=37.5 is extremely smooth — was only +25, now higher penalties
        # Real faces: local variance > 300 (pores, fine lines, texture)
        # AI faces: variance < 80 (too smooth, "painted" / porcelain skin)
        if local_var < 50:
            fake_score += 40
            reasons.append(f"Extremely smooth skin (variance={local_var:.1f}) — no pores or texture, classic AI skin")
        elif local_var < 100:
            fake_score += 25
            reasons.append(f"Unnaturally smooth skin (variance={local_var:.1f}) — insufficient texture detail")
        elif local_var < 200:
            fake_score += 10
            reasons.append(f"Slightly smooth skin texture (variance={local_var:.1f})")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Facial Geometry & Symmetry",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {
                "fake_probability": f"{fake_score:.0f}%",
                "real_probability": f"{real_score:.0f}%",
            },
            "geo_stats": geo_stats,
            "indicators": reasons if reasons else ["Facial geometry looks natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Facial Geometry & Symmetry", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 12: Texture Analysis — Gabor + LBP + Wavelet
# ══════════════════════════════════════════════════════════════
def detect_with_texture_analysis(image_path: str) -> dict:
    """
    Texture Analysis: Gabor Filters + LBP + Wavelet Decomposition

    HOW IT WORKS:
    ─────────────
    Real images have rich, multi-scale, directional texture from the physical world.
    AI images often have texture that is too regular, too smooth, or wrong scale.

    1. GABOR FILTERS — directional texture energy
       Gabor = oriented bandpass filters (like V1 visual cortex neurons)
       Applied at 8 orientations × 4 scales = 32 filter responses
       Real: high energy variance across orientations (anisotropic texture)
       AI:   low variance — unnaturally isotropic (same in all directions)

    2. LBP — Local Binary Pattern
       For each pixel: compare with 8 neighbours → binary code
       Real: rich histogram with natural spread (many unique patterns)
       AI:   LBP histogram too uniform or peaked (repetitive micro-texture)

    3. WAVELET — multi-scale frequency decomposition
       Haar wavelet → detail coefficients at 3 levels
       Real: detail energy decays naturally across scales (1/f noise law)
       AI:   unnatural energy distribution — often too flat or too steep
    """
    try:
        import cv2
        print("[Texture] Running Gabor + LBP + Wavelet analysis...")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        fake_score = 0
        reasons    = []
        tex_stats  = {}

        # ── 1. Gabor filter bank ──
        gabor_energies = []
        for theta in np.linspace(0, np.pi, 8, endpoint=False):
            for sigma in [2, 4, 8, 16]:
                lam  = sigma * 2
                kern = cv2.getGaborKernel((31, 31), sigma, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
                resp = cv2.filter2D(gray, cv2.CV_32F, kern)
                gabor_energies.append(float(np.var(resp)))

        gabor_mean    = float(np.mean(gabor_energies))
        gabor_std     = float(np.std(gabor_energies))
        gabor_isotropy = gabor_std / (gabor_mean + 1e-8)  # high = anisotropic (real)
        tex_stats["gabor_isotropy"] = round(gabor_isotropy, 4)

        if gabor_isotropy < 0.4:
            fake_score += 20
            reasons.append(f"Low Gabor isotropy={gabor_isotropy:.3f} — unnaturally uniform directional texture")

        # ── 2. LBP — Local Binary Pattern ──
        def lbp(img_gray, radius=1, n_points=8):
            h, w   = img_gray.shape
            result = np.zeros((h, w), dtype=np.uint8)
            angles = 2 * np.pi * np.arange(n_points) / n_points
            for i, angle in enumerate(angles):
                dy = int(round(radius * np.sin(angle)))
                dx = int(round(radius * np.cos(angle)))
                shifted = np.roll(np.roll(img_gray, -dy, axis=0), -dx, axis=1)
                result += ((shifted >= img_gray).astype(np.uint8) << i)
            return result

        lbp_img  = lbp(gray.astype(np.uint8))
        lbp_hist, _ = np.histogram(lbp_img.flatten(), bins=256, range=(0, 256), density=True)
        lbp_entropy  = float(-np.sum(lbp_hist[lbp_hist > 0] * np.log2(lbp_hist[lbp_hist > 0] + 1e-10)))
        lbp_uniformity = float(np.sum(lbp_hist ** 2))
        tex_stats["lbp_entropy"]    = round(lbp_entropy, 3)
        tex_stats["lbp_uniformity"] = round(lbp_uniformity, 6)

        # Real: high entropy LBP (varied micro-texture)
        # AI:   low entropy or high uniformity (repetitive texture)
        if lbp_entropy < 5.5:
            fake_score += 20
            reasons.append(f"Low LBP entropy={lbp_entropy:.2f} — micro-texture too repetitive")
        if lbp_uniformity > 0.02:
            fake_score += 15
            reasons.append(f"High LBP uniformity={lbp_uniformity:.5f} — texture pattern too regular")

        # ── 3. Wavelet decomposition ──
        # Manual Haar wavelet using cv2 (no pywt needed)
        def haar_detail_energy(img_f):
            """Return list of detail energies at 3 wavelet levels."""
            energies = []
            current  = img_f.copy()
            for _ in range(3):
                h2, w2 = current.shape[0]//2, current.shape[1]//2
                if h2 < 8 or w2 < 8:
                    break
                # Horizontal detail
                rows_e = (current[::2, :] + current[1::2, :]) / 2
                rows_d = (current[::2, :] - current[1::2, :]) / 2
                # Vertical detail of difference rows
                detail = (rows_d[:, ::2] - rows_d[:, 1::2]) / 2
                energies.append(float(np.mean(detail ** 2)))
                current = (rows_e[:, ::2] + rows_e[:, 1::2]) / 2
            return energies

        wav_energies = haar_detail_energy(gray)
        tex_stats["wavelet_energies"] = [round(e, 4) for e in wav_energies]

        if len(wav_energies) >= 2:
            # Real images: energy decays by ~4-10× per level (1/f noise law)
            decay_ratios = [wav_energies[i] / (wav_energies[i+1] + 1e-8)
                            for i in range(len(wav_energies)-1)]
            avg_decay = float(np.mean(decay_ratios))
            tex_stats["wavelet_decay_ratio"] = round(avg_decay, 3)

            if avg_decay < 1.5:
                fake_score += 20
                reasons.append(f"Flat wavelet energy decay={avg_decay:.2f} — unnatural frequency distribution")
            elif avg_decay > 50:
                fake_score += 15
                reasons.append(f"Extreme wavelet decay={avg_decay:.2f} — over-smoothed at fine scales")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Texture Analysis (Gabor+LBP+Wavelet)",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {"fake_probability": f"{fake_score:.0f}%", "real_probability": f"{real_score:.0f}%"},
            "texture_stats": tex_stats,
            "indicators": reasons if reasons else ["Texture patterns look natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Texture Analysis (Gabor+LBP+Wavelet)", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 13: Geometric Distortion Analysis
# ══════════════════════════════════════════════════════════════
def detect_with_geometric_distortion(image_path: str) -> dict:
    """
    Geometric Distortion Analysis — lens & perspective artifacts

    HOW IT WORKS:
    ─────────────
    Real camera lenses introduce predictable geometric distortions:

    1. BARREL / PINCUSHION DISTORTION
       Wide-angle lenses → barrel distortion (lines curve outward)
       Telephoto lenses  → pincushion distortion (lines curve inward)
       AI images: straight lines are perfectly straight (no lens distortion)

    2. CHROMATIC ABERRATION (geometric)
       Real lenses: R/G/B channels are slightly spatially misaligned
       AI images: channels are perfectly registered (pixel-perfect alignment)

    3. VIGNETTING
       Real lenses: corners slightly darker than center (light falloff)
       AI images: uniform brightness to corners (no vignetting) OR
                  fake vignetting added artificially (too perfect circle)

    4. PERSPECTIVE CONSISTENCY
       Real photos: parallel lines converge toward vanishing points
       AI images: perspective can be inconsistent across the image
    """
    try:
        import cv2
        print("[Geometry] Analyzing geometric distortion patterns...")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w  = gray.shape
        fake_score = 0
        reasons    = []
        geo_stats  = {}

        # ── 1. Vignetting check ──
        # Real lenses: corners darker than center
        cx, cy     = w // 2, h // 2
        r          = min(cx, cy)
        center_bri = float(np.mean(gray[cy-r//4:cy+r//4, cx-r//4:cx+r//4]))

        # Sample corners
        q          = r // 3
        corner_vals = [
            float(np.mean(gray[:q, :q])),           # top-left
            float(np.mean(gray[:q, w-q:])),          # top-right
            float(np.mean(gray[h-q:, :q])),          # bottom-left
            float(np.mean(gray[h-q:, w-q:])),        # bottom-right
        ]
        corner_mean = float(np.mean(corner_vals))
        vignette_ratio = (center_bri - corner_mean) / (center_bri + 1e-8)
        geo_stats["vignette_ratio"] = round(vignette_ratio, 4)

        # Real: slight vignetting (0.02–0.20), AI: near zero or perfect circle
        if vignette_ratio < 0.005:
            fake_score += 20
            reasons.append(f"No vignetting detected (ratio={vignette_ratio:.4f}) — real lenses have corner falloff")
        elif vignette_ratio > 0.35:
            fake_score += 15
            reasons.append(f"Extreme vignetting (ratio={vignette_ratio:.3f}) — possibly artificial post-processing")

        # ── 2. Radial distortion via straight-line detection ──
        edges    = cv2.Canny(gray.astype(np.uint8), 50, 150)
        lines    = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                   minLineLength=w//8, maxLineGap=20)
        geo_stats["hough_lines_found"] = int(len(lines)) if lines is not None else 0

        if lines is not None and len(lines) > 10:
            # Compute angles of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
                angles.append(angle)

            # Real scenes: lines at varied angles (diverse scene geometry)
            # AI scenes: often have suspiciously many perfectly horizontal/vertical lines
            horiz_count = sum(1 for a in angles if a < 5 or a > 175)
            vert_count  = sum(1 for a in angles if 85 < a < 95)
            ortho_ratio = (horiz_count + vert_count) / len(angles)
            geo_stats["orthogonal_line_ratio"] = round(ortho_ratio, 3)

            if ortho_ratio > 0.85:
                fake_score += 20
                reasons.append(f"Too many perfectly orthogonal lines ({ortho_ratio*100:.0f}%) — AI geometry artifact")

        # ── 3. Chromatic aberration spatial check ──
        # Check R vs B channel spatial shift at edges
        r_ch = img[:, :, 2].astype(np.float32)
        b_ch = img[:, :, 0].astype(np.float32)
        edge_mask = edges > 0
        if edge_mask.sum() > 200:
            # Compute local gradient direction agreement between R and B channels
            gr_r = np.gradient(r_ch)
            gr_b = np.gradient(b_ch)
            # Dot product of gradient vectors (high = same direction = no CA)
            dot  = gr_r[0][edge_mask] * gr_b[0][edge_mask] + gr_r[1][edge_mask] * gr_b[1][edge_mask]
            mag_r = np.sqrt(gr_r[0][edge_mask]**2 + gr_r[1][edge_mask]**2) + 1e-8
            mag_b = np.sqrt(gr_b[0][edge_mask]**2 + gr_b[1][edge_mask]**2) + 1e-8
            cos_sim = float(np.mean(dot / (mag_r * mag_b)))
            geo_stats["chromatic_aberration_cos"] = round(cos_sim, 4)

            # Real lenses: cos_sim slightly < 1.0 (slight channel misalignment)
            # AI: cos_sim very close to 1.0 (perfect alignment, no CA)
            if cos_sim > 0.995:
                fake_score += 25
                reasons.append(f"No chromatic aberration (cos={cos_sim:.4f}) — real lenses always show slight CA")

        # ── 4. Radial brightness profile smoothness ──
        # Compute brightness as function of distance from center
        Y, X    = np.mgrid[0:h, 0:w]
        dist    = np.sqrt((X - cx)**2 + (Y - cy)**2).astype(np.float32)
        max_dist= float(dist.max())
        bins    = 20
        radial_means = []
        for i in range(bins):
            d_lo = max_dist * i / bins
            d_hi = max_dist * (i + 1) / bins
            mask = (dist >= d_lo) & (dist < d_hi)
            if mask.sum() > 0:
                radial_means.append(float(np.mean(gray[mask])))

        if len(radial_means) >= 4:
            radial_smoothness = float(np.std(np.diff(radial_means)))
            geo_stats["radial_smoothness"] = round(radial_smoothness, 3)
            if radial_smoothness < 0.3:
                fake_score += 15
                reasons.append(f"Unnaturally smooth radial brightness profile (std={radial_smoothness:.3f})")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Geometric Distortion Analysis",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {"fake_probability": f"{fake_score:.0f}%", "real_probability": f"{real_score:.0f}%"},
            "geo_distortion_stats": geo_stats,
            "indicators": reasons if reasons else ["Geometric distortion looks natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Geometric Distortion Analysis", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 14: Shadow & Highlight Physics
# ══════════════════════════════════════════════════════════════
def detect_with_shadow_physics(image_path: str) -> dict:
    """
    Shadow & Highlight Physics Analysis

    HOW IT WORKS:
    ─────────────
    Real-world lighting creates shadows and highlights that follow physics:

    1. SHADOW CONSISTENCY
       Real photos: shadows are all cast by the same light source
       → Shadow directions are consistent across objects in the scene
       AI images: shadows may come from different directions per object
       → Inconsistent shadow angles = strong fake indicator

    2. SHADOW SOFTNESS vs DISTANCE
       Real physics: soft shadows farther from object, hard near contact point
       AI images: all shadows same softness regardless of distance

    3. HIGHLIGHT POSITION CONSISTENCY
       Real specular highlights: position follows light source angle
       AI images: highlights may appear in physically impossible positions

    4. TONE CURVE NATURALNESS
       Real cameras have characteristic S-curve tone response
       AI images: tone curves can be unnaturally linear or clipped

    5. SHADOW COLOR CAST
       Real outdoor shadows: bluish (lit by sky)
       Real indoor shadows: warm or cool depending on room lighting
       AI shadows: often neutral grey (no color physics)
    """
    try:
        import cv2
        print("[Shadow] Analyzing shadow & highlight physics...")

        img   = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w    = gray.shape

        fake_score = 0
        reasons    = []
        shad_stats = {}

        # ── 1. Shadow mask: dark regions ──
        shadow_mask    = gray < 60
        highlight_mask = gray > 220
        shad_stats["shadow_pct"]    = round(float(np.mean(shadow_mask)) * 100, 2)
        shad_stats["highlight_pct"] = round(float(np.mean(highlight_mask)) * 100, 2)

        # ── 2. Shadow color cast check ──
        # Outdoor real shadows: R < B (sky light = blue)
        # AI shadows: R ≈ G ≈ B (neutral grey, no physics)
        if shadow_mask.sum() > 500:
            r_shad = float(np.mean(img_rgb[:, :, 0][shadow_mask]))
            g_shad = float(np.mean(img_rgb[:, :, 1][shadow_mask]))
            b_shad = float(np.mean(img_rgb[:, :, 2][shadow_mask]))
            shad_color_std = float(np.std([r_shad, g_shad, b_shad]))
            shad_stats["shadow_color_std"] = round(shad_color_std, 3)

            if shad_color_std < 1.5:
                fake_score += 25
                reasons.append(f"Neutral-grey shadows (color_std={shad_color_std:.2f}) — real shadows have color cast from ambient light")

        # ── 3. Highlight clipping check ──
        # Real cameras: smooth rolloff near white (no hard clipping)
        # AI images: sharp clipping at 255 (hard edge in histogram)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        top_bin_ratio = float(hist[255] / (hist[240:255].mean() + 1e-8))
        shad_stats["highlight_clipping_ratio"] = round(top_bin_ratio, 3)

        if top_bin_ratio > 10:
            fake_score += 20
            reasons.append(f"Hard highlight clipping (ratio={top_bin_ratio:.1f}) — AI over-exposes without natural rolloff")

        # ── 4. Tone curve S-shape check ──
        # Real camera: tone curve has slight S-shape (gamma correction)
        # Check via histogram shape across tonal range
        hist_norm  = hist / (hist.sum() + 1e-8)
        cumhist    = np.cumsum(hist_norm)
        # Check midpoint: in natural photos, 50% of pixels below ~128
        # AI images can have weird distributions
        median_tone = int(np.searchsorted(cumhist, 0.5))
        shad_stats["median_tone"] = median_tone

        # Very bright median (>180) or very dark (<60) suggests AI manipulation
        if median_tone > 185 or median_tone < 50:
            fake_score += 15
            reasons.append(f"Unusual median tone={median_tone} — unnatural tonal distribution")

        # ── 5. Shadow gradient direction consistency ──
        # Compute gradient angles in dark regions
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        if shadow_mask.sum() > 200:
            angles_shad = np.degrees(np.arctan2(
                grad_y[shadow_mask], grad_x[shadow_mask]
            ))
            # Real: shadow gradients consistent direction (one light source)
            # AI: random shadow gradients
            angle_std = float(np.std(angles_shad))
            shad_stats["shadow_gradient_std"] = round(angle_std, 2)

            if angle_std < 20:
                # Too consistent = artificially uniform shadow
                fake_score += 20
                reasons.append(f"Unnaturally uniform shadow gradients (std={angle_std:.1f}°) — artificial lighting")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Shadow & Highlight Physics",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {"fake_probability": f"{fake_score:.0f}%", "real_probability": f"{real_score:.0f}%"},
            "shadow_stats": shad_stats,
            "indicators": reasons if reasons else ["Shadow & highlight physics look natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Shadow & Highlight Physics", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 15: Compression Artifact Deep Analysis
# ══════════════════════════════════════════════════════════════
def detect_with_compression_artifacts(image_path: str) -> dict:
    """
    Compression Artifact Deep Analysis

    HOW IT WORKS:
    ─────────────
    JPEG compression creates predictable, measurable artifacts.
    The pattern of these artifacts reveals the image's history.

    1. BLOCKINESS DETECTION (8×8 DCT blocks)
       Real photos saved as JPEG: visible 8×8 block boundaries
       AI images: no blockiness (or artificial blockiness added)
       Method: measure discontinuity at 8-pixel boundaries

    2. RINGING ARTIFACT DETECTION
       JPEG compression causes ringing near sharp edges (Gibbs phenomenon)
       Real: ringing present near high-contrast edges
       AI:   may lack ringing or have wrong ringing pattern

    3. DOUBLE COMPRESSION DETECTION
       Real photos often saved twice (camera → editor → share)
       AI images: often single-compression or no compression history
       Method: analyze DCT coefficient histogram for double-compression signature

    4. MOSQUITO NOISE
       JPEG mosquito noise appears around text and sharp edges
       Real photos: characteristic mosquito noise pattern
       AI images: different or absent mosquito noise
    """
    try:
        import cv2
        print("[Compression] Analyzing compression artifact patterns...")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        fake_score = 0
        reasons    = []
        comp_stats = {}

        # ── 1. Blockiness at 8-pixel boundaries ──
        # Real JPEG: discontinuity at 8px boundaries
        # Measure average difference at block boundaries vs interior
        block_diffs  = []
        interior_diffs = []
        for x in range(0, w - 1, 1):
            col_diff = float(np.mean(np.abs(gray[:, x].astype(float) - gray[:, x+1].astype(float))))
            if x % 8 == 7:
                block_diffs.append(col_diff)
            else:
                interior_diffs.append(col_diff)

        blockiness_ratio = (np.mean(block_diffs) / (np.mean(interior_diffs) + 1e-8)
                            if block_diffs and interior_diffs else 1.0)
        comp_stats["blockiness_ratio"] = round(float(blockiness_ratio), 4)

        # Real JPEG: blockiness_ratio > 1.05 (blocks visible at boundaries)
        # AI or PNG: ratio ≈ 1.0 (no block structure)
        if blockiness_ratio < 1.02:
            fake_score += 25
            reasons.append(f"No JPEG block structure (ratio={blockiness_ratio:.4f}) — AI images lack natural compression history")

        # ── 2. DCT coefficient histogram analysis ──
        # Analyze 8×8 DCT blocks across the image
        dct_ac_coeffs = []
        step = 8
        for i in range(0, h - step, step):
            for j in range(0, w - step, step):
                block = gray[i:i+step, j:j+step]
                dct   = cv2.dct(block.copy())
                # AC coefficients (exclude DC at [0,0])
                ac    = dct.flatten()[1:]
                dct_ac_coeffs.extend(ac.tolist())

        dct_ac = np.array(dct_ac_coeffs, dtype=np.float32)
        # Quantization signature: real JPEG → DCT coefficients cluster at multiples of quantization step
        dct_hist, dct_bins = np.histogram(dct_ac, bins=200, range=(-50, 50))
        # Count zero-coefficient ratio (JPEG quantization creates many exact zeros)
        zero_ratio = float(np.sum(np.abs(dct_ac) < 0.5) / len(dct_ac))
        comp_stats["dct_zero_ratio"] = round(zero_ratio, 4)

        # Real JPEG: zero_ratio typically 0.30–0.65
        # AI image saved as PNG then JPEG: zero_ratio too high (>0.75) or too low (<0.20)
        if zero_ratio > 0.75:
            fake_score += 20
            reasons.append(f"Extreme DCT sparsity (zero_ratio={zero_ratio:.3f}) — over-quantized or AI-generated")
        elif zero_ratio < 0.20:
            fake_score += 15
            reasons.append(f"Low DCT sparsity (zero_ratio={zero_ratio:.3f}) — unusual for JPEG-compressed image")

        # ── 3. Ringing artifact check ──
        # Detect Gibbs phenomenon near edges
        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
        ring_zone = (dilated_edges > 0) & (edges == 0)

        if ring_zone.sum() > 100:
            ring_variance = float(np.var(gray[ring_zone]))
            comp_stats["ringing_variance"] = round(ring_variance, 3)
            # Real JPEG near edges: moderate ringing variance (20–200)
            if ring_variance < 5:
                fake_score += 20
                reasons.append(f"No ringing artifacts near edges (var={ring_variance:.1f}) — JPEG ringing absent")

        # ── 4. Mosquito noise ──
        # High-pass filter to isolate noise near edges
        blurred  = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray - blurred
        near_edge_noise = float(np.std(residual[dilated_edges > 0])) if (dilated_edges > 0).sum() > 100 else 0
        far_noise       = float(np.std(residual[dilated_edges == 0])) if (dilated_edges == 0).sum() > 100 else 0
        mosquito_ratio  = near_edge_noise / (far_noise + 1e-8)
        comp_stats["mosquito_noise_ratio"] = round(mosquito_ratio, 3)

        # Real JPEG: more noise near edges (mosquito effect) → ratio > 1.3
        if mosquito_ratio < 1.1:
            fake_score += 15
            reasons.append(f"Low mosquito noise ratio={mosquito_ratio:.3f} — real JPEG has more noise near edges")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Compression Artifact Analysis",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {"fake_probability": f"{fake_score:.0f}%", "real_probability": f"{real_score:.0f}%"},
            "compression_stats": comp_stats,
            "indicators": reasons if reasons else ["Compression artifacts look natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Compression Artifact Analysis", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 16: Reflection & Refraction Physics
# ══════════════════════════════════════════════════════════════
def detect_with_reflection_physics(image_path: str) -> dict:
    """
    Reflection & Refraction Physics Analysis

    HOW IT WORKS:
    ─────────────
    Real scenes obey the laws of optics. AI generators often violate them:

    1. SPECULAR REFLECTION CONSISTENCY
       Real: specular highlights appear at angle of incidence = angle of reflection
       AI:   highlights appear in physically impossible positions

    2. FRESNEL EFFECT
       Real: reflectivity increases at grazing angles (near edges of curved surfaces)
       AI:   uniform reflectivity regardless of surface angle

    3. REFLECTION SYMMETRY IN WATER/GLASS
       Real water reflections: vertically flipped, slightly blurred, wavy
       AI water: may have imperfect or asymmetric reflections

    4. SHADOW-HIGHLIGHT PAIRING
       Real: every highlight implies a corresponding shadow direction
       AI:   highlights without corresponding shadows

    We detect these via:
      - Bright region analysis (specular highlights)
      - Symmetry of bright regions across potential reflecting surfaces
      - Highlight shape (real = roughly circular/elliptical)
      - Gradient analysis around highlight edges
    """
    try:
        import cv2
        print("[Reflection] Analyzing reflection & refraction physics...")

        img  = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        fake_score = 0
        reasons    = []
        refl_stats = {}

        # ── 1. Specular highlight detection ──
        _, spec_mask = cv2.threshold(gray.astype(np.uint8), 230, 255, cv2.THRESH_BINARY)
        spec_count   = int(np.sum(spec_mask > 0))
        refl_stats["specular_pixel_count"] = spec_count

        if spec_count > 100:
            # ── 2. Highlight shape analysis (should be roughly elliptical) ──
            contours, _ = cv2.findContours(spec_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            circularities = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                if area > 20 and peri > 0:
                    circ = 4 * np.pi * area / (peri ** 2)
                    circularities.append(float(circ))

            if circularities:
                mean_circ = float(np.mean(circularities))
                refl_stats["highlight_circularity"] = round(mean_circ, 3)
                # Real specular: circularity 0.5–1.0
                # AI specular: can be oddly shaped (< 0.3) or too perfect (1.0)
                if mean_circ < 0.3:
                    fake_score += 20
                    reasons.append(f"Irregular specular shape (circ={mean_circ:.3f}) — real highlights are elliptical")

            # ── 3. Highlight gradient softness ──
            # Real highlights: soft gradient at edges (Fresnel rolloff)
            # AI highlights: can have hard edges
            highlight_dil  = cv2.dilate(spec_mask, np.ones((7, 7), np.uint8))
            highlight_ring = (highlight_dil > 0) & (spec_mask == 0)
            if highlight_ring.sum() > 50:
                ring_gradient = float(np.mean(np.abs(
                    cv2.Sobel(gray, cv2.CV_32F, 1, 0)[highlight_ring]
                )))
                refl_stats["highlight_edge_gradient"] = round(ring_gradient, 3)
                if ring_gradient > 80:
                    fake_score += 20
                    reasons.append(f"Hard highlight edges (grad={ring_gradient:.1f}) — real Fresnel effect gives soft rolloff")

        # ── 4. Reflection symmetry check ──
        # Check if image has vertical reflection (water surface)
        top_half    = gray[:h//2, :]
        bottom_half = cv2.flip(gray[h//2:, :], 0)
        min_h2      = min(top_half.shape[0], bottom_half.shape[0])
        top_half    = top_half[:min_h2, :]
        bottom_half = bottom_half[:min_h2, :]
        vert_sym    = 1.0 - float(np.mean(np.abs(top_half - bottom_half))) / 128.0
        refl_stats["vertical_symmetry"] = round(vert_sym, 4)

        # Perfect vertical symmetry = suspicious (AI reflection artifact)
        if vert_sym > 0.92:
            fake_score += 25
            reasons.append(f"Suspiciously perfect vertical reflection (sym={vert_sym:.3f}) — real reflections are imperfect")

        # ── 5. Highlight count vs scene brightness ──
        mean_brightness = float(np.mean(gray))
        expected_spec_ratio = mean_brightness / 255.0
        actual_spec_ratio   = spec_count / (h * w)
        spec_excess         = actual_spec_ratio / (expected_spec_ratio + 1e-8)
        refl_stats["specular_excess"] = round(spec_excess, 3)

        if spec_excess > 5.0:
            fake_score += 15
            reasons.append(f"Excess specular highlights (excess={spec_excess:.1f}×) — overlit AI artifact")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Reflection & Refraction Physics",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {"fake_probability": f"{fake_score:.0f}%", "real_probability": f"{real_score:.0f}%"},
            "reflection_stats": refl_stats,
            "indicators": reasons if reasons else ["Reflection physics look natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Reflection & Refraction Physics", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 17: Background Blur Realism (Bokeh Analysis)
# ══════════════════════════════════════════════════════════════
def detect_with_bokeh_analysis(image_path: str) -> dict:
    """
    Background Blur Realism — Bokeh Analysis

    HOW IT WORKS:
    ─────────────
    Real camera bokeh (background blur) has specific optical characteristics:

    1. DEPTH-CONSISTENT BLUR GRADIENT
       Real: blur increases smoothly with distance from focus plane
       AI:   blur may jump abruptly or be applied uniformly

    2. BOKEH SHAPE (aperture shape)
       Real lenses: bokeh circles are shaped by aperture blades
         (hexagonal for 6-blade, circular for round aperture)
       AI: bokeh shapes may be perfect circles (too ideal) or wrong shape

    3. BLUR-EDGE BOUNDARY SHARPNESS
       Real depth of field: sharp foreground transitions to blur at natural depth
       AI: foreground-background boundary can be unnaturally sharp ("cutout" effect)

    4. CHROMATIC BOKEH
       Real lenses: bokeh circles have chromatic fringing (green/purple edges)
       AI bokeh: spectrally neutral, no chromatic fringing

    5. FOCUS PLANE CONSISTENCY
       Real: one continuous focus plane visible in image
       AI: may have multiple focus planes or inconsistent depth

    FIX v6.1: Large images (>1920px) are resized to max 1920px before
    processing to avoid OpenCV filter memory errors on 4K+ images.
    """
    try:
        import cv2
        print("[Bokeh] Analyzing background blur realism...")

        img_orig = cv2.imread(image_path)
        if img_orig is None:
            raise ValueError("Cannot read image")

        # ── Safe resize: cap at 1920px wide to avoid OOM on large images ──
        MAX_DIM = 1920
        oh, ow  = img_orig.shape[:2]
        if max(oh, ow) > MAX_DIM:
            scale = MAX_DIM / max(oh, ow)
            img   = cv2.resize(img_orig, (int(ow * scale), int(oh * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            img = img_orig

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        fake_score  = 0
        reasons     = []
        bokeh_stats = {}

        # ── 1. Local blur map ──
        # Laplacian variance in local windows = local sharpness map
        win       = 32
        sharp_map = np.zeros((h // win, w // win), dtype=np.float32)
        for i in range(sharp_map.shape[0]):
            for j in range(sharp_map.shape[1]):
                patch = gray[i*win:(i+1)*win, j*win:(j+1)*win]
                sharp_map[i, j] = float(cv2.Laplacian(patch, cv2.CV_64F).var())

        # ── 2. Blur gradient check ──
        sharp_gradient = float(np.std(np.diff(sharp_map, axis=0)))
        bokeh_stats["sharpness_gradient_std"] = round(sharp_gradient, 3)
        sharp_range = float(np.max(sharp_map) - np.min(sharp_map))
        bokeh_stats["sharpness_range"] = round(sharp_range, 1)

        has_bokeh = sharp_range > 500

        if has_bokeh:
            if sharp_gradient < 2.0:
                fake_score += 25
                reasons.append(f"Abrupt bokeh boundary (gradient_std={sharp_gradient:.2f}) — AI depth-of-field is not gradual")

            focus_mask      = sharp_map > np.percentile(sharp_map, 75)
            blur_mask       = sharp_map < np.percentile(sharp_map, 25)
            focus_sharpness = float(np.mean(sharp_map[focus_mask]))
            blur_sharpness  = float(np.mean(sharp_map[blur_mask]))
            bokeh_ratio     = focus_sharpness / (blur_sharpness + 1e-8)
            bokeh_stats["focus_to_blur_ratio"] = round(bokeh_ratio, 2)

            if bokeh_ratio > 100:
                fake_score += 25
                reasons.append(f"Unnatural bokeh strength (ratio={bokeh_ratio:.1f}×) — AI portrait mode artifact")

        # ── 3. Chromatic bokeh check ──
        blur_region_mask = cv2.resize(
            (sharp_map < np.percentile(sharp_map, 30)).astype(np.uint8),
            (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        if blur_region_mask.sum() > 1000:
            r_ch = img[:, :, 2].astype(np.float32)
            g_ch = img[:, :, 1].astype(np.float32)
            rg_diff_blur = float(np.std(r_ch[blur_region_mask] - g_ch[blur_region_mask]))
            bokeh_stats["bokeh_chromatic_std"] = round(rg_diff_blur, 3)

            if rg_diff_blur < 1.5 and has_bokeh:
                fake_score += 20
                reasons.append(f"No chromatic fringing in bokeh (std={rg_diff_blur:.2f}) — real lenses show slight CA")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Background Blur / Bokeh Analysis",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {"fake_probability": f"{fake_score:.0f}%", "real_probability": f"{real_score:.0f}%"},
            "bokeh_stats": bokeh_stats,
            "indicators": reasons if reasons else ["Bokeh/blur realism looks natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Background Blur / Bokeh Analysis", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 18: Object Boundary Sharpness Anomaly
# ══════════════════════════════════════════════════════════════
def detect_with_boundary_analysis(image_path: str) -> dict:
    """
    Object Boundary Sharpness Anomaly Detection

    HOW IT WORKS:
    ─────────────
    AI generators often produce inconsistent sharpness at object boundaries:

    1. SHARPNESS CONSISTENCY AT EDGES
       Real photos: edge sharpness consistent with depth and motion
       AI images: some edges unnaturally sharp, others oddly blurry

    2. RINGING AT BOUNDARIES
       Real: slight ringing at high-contrast edges (JPEG + optics)
       AI:   may have excessive ringing (upscaling artifacts) or none

    3. TEXTURE-EDGE CORRELATION
       Real: texture inside object matches edge sharpness level
       AI:   sharp edges with no interior texture = classic "AI smooth" artifact

    4. BOUNDARY HALO DETECTION
       AI images often produce bright/dark halos at object-background boundaries

    5. EDGE DIRECTION CONSISTENCY
       Real scenes: edge directions follow 3D scene structure
       AI: may have locally inconsistent edge directions

    FIX v6.1: Large images resized to max 1920px; local_sharpness loop
    replaced with fast cv2.boxFilter approach to avoid OOM on 4K images.
    """
    try:
        import cv2
        print("[Boundary] Analyzing object boundary sharpness...")

        img_orig = cv2.imread(image_path)
        if img_orig is None:
            raise ValueError("Cannot read image")

        # ── Safe resize: cap at 1920px to avoid OOM on large images ──
        MAX_DIM = 1920
        oh, ow  = img_orig.shape[:2]
        if max(oh, ow) > MAX_DIM:
            scale = MAX_DIM / max(oh, ow)
            img   = cv2.resize(img_orig, (int(ow * scale), int(oh * scale)),
                               interpolation=cv2.INTER_AREA)
        else:
            img = img_orig

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape

        fake_score  = 0
        reasons     = []
        bound_stats = {}

        # ── 1. Edge gradient magnitude distribution ──
        grad_x   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        edge_mask  = grad_mag > np.percentile(grad_mag, 90)
        edge_grads = grad_mag[edge_mask]
        grad_mean  = float(np.mean(edge_grads))
        grad_std   = float(np.std(edge_grads))
        grad_cv    = grad_std / (grad_mean + 1e-8)
        bound_stats["edge_gradient_mean"] = round(grad_mean, 2)
        bound_stats["edge_gradient_cv"]   = round(grad_cv, 4)

        if grad_cv < 0.3:
            fake_score += 20
            reasons.append(f"Uniform edge strength (CV={grad_cv:.3f}) — real images have varied edge intensities")

        # ── 2. Halo detection ──
        edges_bin = cv2.Canny(gray.astype(np.uint8), 50, 150)
        k5        = np.ones((5, 5), np.uint8)
        k9        = np.ones((9, 9), np.uint8)
        dilated   = cv2.dilate(edges_bin, k5)
        halo_zone = (dilated > 0) & (edges_bin == 0)

        if halo_zone.sum() > 100:
            surrounding   = cv2.dilate(dilated, k9)
            surround_zone = (surrounding > 0) & (dilated == 0)
            halo_brightness     = float(np.mean(gray[halo_zone]))
            surround_brightness = float(np.mean(gray[surround_zone])) if surround_zone.sum() > 0 else halo_brightness
            halo_excess = abs(halo_brightness - surround_brightness)
            bound_stats["halo_excess"] = round(halo_excess, 3)

            if halo_excess > 15:
                fake_score += 25
                reasons.append(f"Boundary halo detected (excess={halo_excess:.1f}) — AI upscaling/generation artifact")

        # ── 3. Texture-edge correlation (fast version using integral image) ──
        # Compute local Laplacian variance map using sliding window via boxFilter
        lap         = cv2.Laplacian(gray, cv2.CV_32F)
        lap_sq      = lap ** 2
        win         = 16
        # mean of lap in window
        lap_mean    = cv2.boxFilter(lap,    cv2.CV_32F, (win, win))
        lap_sq_mean = cv2.boxFilter(lap_sq, cv2.CV_32F, (win, win))
        local_var   = np.maximum(lap_sq_mean - lap_mean**2, 0)  # variance = E[X²] - E[X]²

        sharp_at_edges     = float(np.mean(local_var[edge_mask]))
        sharp_not_at_edges = float(np.mean(local_var[~edge_mask]))
        edge_interior_ratio = sharp_at_edges / (sharp_not_at_edges + 1e-8)
        bound_stats["edge_interior_sharpness_ratio"] = round(edge_interior_ratio, 3)

        if edge_interior_ratio > 30:
            fake_score += 25
            reasons.append(f"Sharp edges with smooth interior (ratio={edge_interior_ratio:.1f}) — classic AI texture artifact")

        # ── 4. Edge direction entropy ──
        grad_angles = np.degrees(np.arctan2(grad_y[edge_mask], grad_x[edge_mask])) % 180
        angle_hist, _ = np.histogram(grad_angles, bins=36, range=(0, 180))
        angle_hist_norm = angle_hist / (angle_hist.sum() + 1e-8)
        direction_entropy = float(-np.sum(
            angle_hist_norm[angle_hist_norm > 0] *
            np.log2(angle_hist_norm[angle_hist_norm > 0] + 1e-10)
        ))
        bound_stats["edge_direction_entropy"] = round(direction_entropy, 3)

        if direction_entropy < 3.5:
            fake_score += 10
            reasons.append(f"Low edge direction diversity (entropy={direction_entropy:.2f}) — unnatural geometry")

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Object Boundary Sharpness",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {"fake_probability": f"{fake_score:.0f}%", "real_probability": f"{real_score:.0f}%"},
            "boundary_stats": bound_stats,
            "indicators": reasons if reasons else ["Object boundaries look natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Object Boundary Sharpness", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# APPROACH 19: Face Landmark 3D Consistency
# ══════════════════════════════════════════════════════════════
def detect_with_face_3d_consistency(image_path: str) -> dict:
    """
    Face Landmark 3D Consistency Analysis

    HOW IT WORKS:
    ─────────────
    AI-generated faces often have subtle 3D inconsistencies
    that are not visible to casual inspection but measurable:

    1. EAR VISIBILITY CONSISTENCY
       Real face at 3/4 view: one ear visible, other hidden
       AI faces: both ears sometimes visible at impossible angles

    2. FACIAL REGION PROPORTION
       Real faces follow anthropometric ratios (forensic anthropology):
         - Eye width ≈ nose width
         - Face width at eyes ≈ 5× eye width
         - Upper lip to nose ≈ lower lip to chin
       AI faces: subtle proportion violations

    3. SKIN TONE CONSISTENCY ACROSS FACE
       Real: skin tone varies slightly across face (nose brighter,
             under-eyes darker, cheeks have more red)
       AI:   skin tone unnaturally uniform (one-pass generation)

    4. FACIAL REGION TEXTURE GRADIENT
       Real: different facial regions have different texture levels
             (pores on cheeks, smooth forehead, textured lips)
       AI:   uniform texture level across all facial regions

    Uses OpenCV Haar cascades + face region analysis
    (no external face landmark model needed)
    """
    try:
        import cv2
        print("[Face3D] Analyzing face 3D consistency...")

        img  = cv2.imread(image_path)
        if img is None:
            raise ValueError("Cannot read image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        face_cascade    = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            return {
                "method": "Face 3D Consistency",
                "prediction": "REAL",
                "confidence": "50%",
                "scores": {"fake_probability": "50%", "real_probability": "50%"},
                "indicators": ["No frontal face detected — 3D consistency check skipped"],
                "success": True
            }

        fake_score  = 0
        reasons     = []
        face3d_stats = {}
        face3d_stats["faces_detected"] = int(len(faces))

        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_gray    = gray[y:y+fh, x:x+fw]
        face_rgb     = img_rgb[y:y+fh, x:x+fw]
        face3d_stats["face_size"] = f"{fw}×{fh}"

        # ── 1. Skin tone consistency across facial regions ──
        # Divide face into 9 regions (3×3 grid)
        rh, rw = fh // 3, fw // 3
        region_means = []
        for ri in range(3):
            for rj in range(3):
                region = face_rgb[ri*rh:(ri+1)*rh, rj*rw:(rj+1)*rw]
                region_means.append(float(np.mean(region)))

        skin_tone_std = float(np.std(region_means))
        face3d_stats["skin_tone_region_std"] = round(skin_tone_std, 3)

        # Real: skin tone varies across regions (std > 8)
        # AI: unnaturally uniform skin tone (std < 4)
        if skin_tone_std < 4.0:
            fake_score += 30
            reasons.append(f"Uniform skin tone across face regions (std={skin_tone_std:.2f}) — AI generates flat skin")

        # ── 2. Facial region texture gradient ──
        # Real: upper face (forehead) smoother than lower face (cheeks/chin)
        upper_texture = float(cv2.Laplacian(face_gray[:fh//3, :], cv2.CV_64F).var())
        lower_texture = float(cv2.Laplacian(face_gray[2*fh//3:, :], cv2.CV_64F).var())
        texture_ratio = upper_texture / (lower_texture + 1e-8)
        face3d_stats["upper_lower_texture_ratio"] = round(texture_ratio, 4)

        # Real: lower face often more textured (stubble, pores) → ratio < 1
        # or similar — but should NOT be exactly equal
        if abs(texture_ratio - 1.0) < 0.05:
            fake_score += 20
            reasons.append(f"Identical texture upper/lower face (ratio={texture_ratio:.3f}) — AI generates uniform texture")

        # ── 3. Red channel face gradient ──
        # Real faces: cheek area has more red (blood vessels)
        # AI faces: red channel uniform
        r_ch = face_rgb[:, :, 0]
        left_cheek  = float(np.mean(r_ch[fh//4:3*fh//4, :fw//3]))
        right_cheek = float(np.mean(r_ch[fh//4:3*fh//4, 2*fw//3:]))
        center_face = float(np.mean(r_ch[fh//4:3*fh//4, fw//3:2*fw//3]))
        cheek_red_diff = abs(left_cheek - center_face) + abs(right_cheek - center_face)
        face3d_stats["cheek_red_variation"] = round(cheek_red_diff, 3)

        if cheek_red_diff < 2.0:
            fake_score += 25
            reasons.append(f"No cheek red variation (diff={cheek_red_diff:.2f}) — real faces have redness in cheek areas")

        # ── 4. Facial width-to-height ratio ──
        # Normal face: width/height ≈ 0.6–0.8
        face_ratio = fw / (fh + 1e-8)
        face3d_stats["face_aspect_ratio"] = round(face_ratio, 3)
        if face_ratio > 1.1 or face_ratio < 0.4:
            fake_score += 15
            reasons.append(f"Unusual face aspect ratio ({face_ratio:.2f}) — proportions outside normal range")

        # ── 5. Profile face detection (ears) ──
        # If frontal face detected, profile face should NOT be detected
        # (would indicate impossible angle)
        profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))
        if len(profiles) > 0 and len(faces) > 0:
            face3d_stats["both_frontal_and_profile"] = True
            # Check overlap between frontal and profile detections
            fx, fy, fw2, fh2 = faces[0]
            for px, py, pw, ph in profiles:
                overlap_x = max(0, min(fx+fw2, px+pw) - max(fx, px))
                overlap_y = max(0, min(fy+fh2, py+ph) - max(fy, py))
                if overlap_x * overlap_y > 0.3 * fw2 * fh2:
                    fake_score += 20
                    reasons.append("Both frontal and profile face detected in same region — physically impossible angle")
                    break

        fake_score = min(fake_score, 100)
        real_score = 100 - fake_score
        prediction = "FAKE (AI-Generated)" if fake_score >= 50 else "REAL"

        return {
            "method": "Face 3D Consistency",
            "prediction": prediction,
            "confidence": f"{max(fake_score, real_score):.0f}%",
            "scores": {"fake_probability": f"{fake_score:.0f}%", "real_probability": f"{real_score:.0f}%"},
            "face3d_stats": face3d_stats,
            "indicators": reasons if reasons else ["Face 3D consistency looks natural (REAL)"],
            "success": True
        }
    except Exception as e:
        return {"method": "Face 3D Consistency", "success": False, "error": str(e)}


# ══════════════════════════════════════════════════════════════
# ENSEMBLE: Weighted majority voting
# ══════════════════════════════════════════════════════════════
def ensemble_predict(results: list) -> dict:
    """
    Confidence-weighted ensemble with outlier detection & dynamic weight adjustment.

    FIX 1 — Outlier Penalty:
      If a method votes against the majority with high confidence,
      but the majority is also highly confident, the outlier's weight
      is penalized. This prevents one wrong high-confidence method
      (e.g. HuggingFace 95.8% REAL when 7 others say FAKE) from
      dominating the ensemble.

    FIX 2 — Strong Signal Bonus:
      If SDXL or Deepfake detector says FAKE with > 90% confidence,
      that is a very strong forensic signal — give a bonus multiplier.

    FIX 3 — Hard Override Rules:
      Certain combinations of evidence are so strong that the verdict
      is forced regardless of other votes:
        • EXIF missing + Noise physics violation + SDXL/Deepfake FAKE
          → Force FAKE override
        • 3+ methods with > 80% confidence all agree on FAKE
          → Boost fake score significantly

    Base weights:
      ★★★★★  HuggingFace, Deepfake Detector    (5)
      ★★★★   ELA, Noise/PRNU                   (4)
      ★★★    EXIF, SDXL, Color, Face           (3)
      ★★     CNN, Frequency                    (2)
      ★      CLIP                              (1)
    """
    base_weights = {
        # ── Forensics-grade ML models ──────────────────────── ★★★★★
        "HuggingFace AI Detector":                   5,
        "Deepfake/GAN Detector (prithivMLmods)":     5,
        # ── Physics-based forensic signals ─────────────────── ★★★★
        "ELA (Error Level Analysis)":                4,
        "Noise/Sensor Pattern Analysis (PRNU)":      4,
        "Compression Artifact Analysis":             4,
        # ── Domain-specific ML models ───────────────────────── ★★★
        "SDXL Detector (Organika)":                  3,
        "EXIF Metadata Analysis":                    3,
        "Shadow & Highlight Physics":                3,
        "Geometric Distortion Analysis":             3,
        "Texture Analysis (Gabor+LBP+Wavelet)":      3,
        # ── Optical & structural analysis ───────────────────── ★★★
        "Color & Lighting Physics":                  3,
        "Reflection & Refraction Physics":           3,
        "Background Blur / Bokeh Analysis":          3,
        "Object Boundary Sharpness":                 3,
        # ── Face analysis ───────────────────────────────────── ★★★
        "Facial Geometry & Symmetry":                3,
        "Face 3D Consistency":                       3,
        # ── General feature analysis ────────────────────────── ★★
        "CNN Feature Analysis (ResNet50)":           2,
        "Frequency/Statistical Analysis (FFT+DCT)":  2,
        # ── Zero-shot ───────────────────────────────────────── ★
        "CLIP Zero-Shot":                            1,
    }

    successful = [r for r in results if r.get("success")]
    if not successful:
        return {"final_verdict": "UNABLE TO DETERMINE", "reason": "All methods failed"}

    # ── Pass 1: Count raw votes to find majority direction ──
    raw_fake_votes = sum(1 for r in successful if "FAKE" in r.get("prediction", ""))
    raw_real_votes = len(successful) - raw_fake_votes
    majority_is_fake = raw_fake_votes > raw_real_votes

    # ── Pass 2: Compute contributions with outlier penalty ──
    fake_score     = 0.0
    real_score     = 0.0
    method_breakdown = []

    # High-confidence methods voting against a strong majority get penalized
    strong_majority = max(raw_fake_votes, raw_real_votes) >= (len(successful) * 0.65)

    for r in successful:
        base_w   = base_weights.get(r["method"], 1)
        conf_str = r.get("confidence", "50%").replace("%", "")
        try:
            conf = float(conf_str) / 100.0
        except:
            conf = 0.5

        is_fake    = "FAKE" in r.get("prediction", "")
        is_outlier = (majority_is_fake and not is_fake) or (not majority_is_fake and is_fake)

        # ── Outlier penalty ──
        # If strong majority exists AND this method disagrees with high confidence
        # → reduce its effective weight (it's likely the wrong one)
        adjusted_w = base_w
        penalty_note = ""
        if strong_majority and is_outlier and conf > 0.75:
            # Scale penalty: 95% confident outlier → 40% weight reduction
            penalty_factor = 1.0 - (conf - 0.5) * 0.8
            adjusted_w = base_w * max(penalty_factor, 0.3)
            penalty_note = f" [outlier penalty ×{penalty_factor:.2f}]"

        # ── Strong signal bonus ──
        # SDXL or Deepfake detector with > 90% confidence → bonus
        bonus_note = ""
        if r["method"] in ("SDXL Detector (Organika)",
                           "Deepfake/GAN Detector (prithivMLmods)") and conf > 0.90:
            adjusted_w *= 1.5
            bonus_note = " [high-conf forensic bonus ×1.5]"

        contribution = adjusted_w * conf

        if is_fake:
            fake_score += contribution
        else:
            real_score += contribution

        method_breakdown.append({
            "method":       r["method"],
            "prediction":   "FAKE" if is_fake else "REAL",
            "confidence":   f"{conf*100:.1f}%",
            "base_weight":  base_w,
            "adj_weight":   round(adjusted_w, 2),
            "contribution": round(contribution, 2),
            "direction":    "→ FAKE" if is_fake else "→ REAL",
            "note":         (penalty_note + bonus_note).strip(),
        })

    # ── Fix 3: Hard Override Rules ──
    override_reason = None

    # Rule A: Forensic triad — EXIF missing + Noise violation + specialist model FAKE
    exif_result    = next((r for r in successful if r["method"] == "EXIF Metadata Analysis"), None)
    noise_result   = next((r for r in successful if r["method"] == "Noise/Sensor Pattern Analysis (PRNU)"), None)
    sdxl_result    = next((r for r in successful if r["method"] == "SDXL Detector (Organika)"), None)
    deepfake_result= next((r for r in successful if r["method"] == "Deepfake/GAN Detector (prithivMLmods)"), None)

    exif_fake    = exif_result    and "FAKE" in exif_result.get("prediction", "")
    noise_fake   = noise_result   and "FAKE" in noise_result.get("prediction", "")
    sdxl_fake    = sdxl_result    and "FAKE" in sdxl_result.get("prediction", "")
    deepfake_fake= deepfake_result and "FAKE" in deepfake_result.get("prediction", "")

    if exif_fake and noise_fake and (sdxl_fake or deepfake_fake):
        # Force FAKE — forensic triad confirmed
        fake_score  = max(fake_score, real_score * 2.5)
        override_reason = "⚡ OVERRIDE: Forensic triad — EXIF missing + Noise physics violation + Specialist model FAKE"

    # Rule B: 3+ high-confidence methods (>80%) all say FAKE
    high_conf_fake = [
        r for r in successful
        if "FAKE" in r.get("prediction", "")
        and float(r.get("confidence","0%").replace("%","")) > 80
    ]
    if len(high_conf_fake) >= 3:
        fake_score  = max(fake_score, real_score * 2.0)
        rule_b_note = f"⚡ OVERRIDE: {len(high_conf_fake)} methods with >80% confidence all say FAKE"
        override_reason = override_reason or rule_b_note

    total    = fake_score + real_score
    fake_pct = fake_score / total * 100
    real_pct = real_score / total * 100
    verdict  = "🤖 FAKE (AI-Generated)" if fake_score > real_score else "📷 REAL"

    return {
        "final_verdict":   verdict,
        "confidence":      f"{max(fake_pct, real_pct):.1f}%",
        "weighted_scores": {
            "fake": f"{fake_pct:.1f}%",
            "real": f"{real_pct:.1f}%",
        },
        "method_breakdown":  method_breakdown,
        "override_triggered": override_reason,
        "raw_votes":         {"fake": raw_fake_votes, "real": raw_real_votes},
        "methods_used":      len(successful),
        "methods_failed":    len(results) - len(successful),
    }


# ══════════════════════════════════════════════════════════════
# MAIN — analyze_image()
# ══════════════════════════════════════════════════════════════
def analyze_image(image_path: str):
    """Run all 19 detection methods + save ELA visualization."""
    print("\n" + "═"*70)
    print("  🔍  FAKE vs REAL IMAGE DETECTOR  v6.0")
    print("  19 Methods: Forensic · Physics · Texture · Face · Deep Learning")
    print("═"*70)
    print(f"  📁  Image : {image_path}")

    if not os.path.exists(image_path):
        print(f"\n  ❌  File not found: {image_path}")
        return None

    try:
        img = Image.open(image_path)
        print(f"  📐  Size  : {img.size[0]} × {img.size[1]} px  |  Mode: {img.mode}")
    except Exception as e:
        print(f"\n  ❌  Cannot open image: {e}")
        return None

    print("\n  Running 19 detection methods...\n")

    all_methods = [
        # ── Forensic signals ──
        ("🔬 ELA  (Error Level Analysis)",              detect_with_ela),
        ("📷 Noise/Sensor Pattern  (PRNU)",             detect_with_noise_analysis),
        ("🗂️  EXIF Metadata Analysis",                   detect_with_exif),
        ("🗜️  Compression Artifact Analysis",            detect_with_compression_artifacts),
        # ── Physics & optics ──
        ("🎨 Color & Lighting Physics",                  detect_with_color_physics),
        ("🌑 Shadow & Highlight Physics",                detect_with_shadow_physics),
        ("📐 Geometric Distortion Analysis",             detect_with_geometric_distortion),
        ("💡 Reflection & Refraction Physics",           detect_with_reflection_physics),
        ("🔭 Background Blur / Bokeh Analysis",          detect_with_bokeh_analysis),
        ("🔲 Object Boundary Sharpness",                 detect_with_boundary_analysis),
        # ── Texture & pattern ──
        ("🧩 Texture Analysis (Gabor+LBP+Wavelet)",     detect_with_texture_analysis),
        ("📊 Frequency Analysis (FFT + DCT)",            detect_with_frequency_analysis),
        # ── Face analysis ──
        ("👤 Facial Geometry & Symmetry",                detect_with_facial_geometry),
        ("🧠 Face 3D Consistency",                       detect_with_face_3d_consistency),
        # ── Deep learning ──
        ("🤖 CNN  (ResNet50 Feature Analysis)",          detect_with_cnn),
        ("🤗 HuggingFace AI Detector",                   detect_with_hf_model),
        ("🎨 SDXL Detector (Organika)",                  detect_with_sdxl_detector),
        ("👾 Deepfake/GAN Detector (prithivMLmods)",     detect_with_deepfake_detector),
        ("✂️  CLIP Zero-Shot Classification",             detect_with_clip),
    ]

    results     = []
    ela_vis_path = None

    for display_name, method_fn in all_methods:
        print(f"  ── {display_name}")
        result = method_fn(image_path)
        results.append(result)

        # Capture ELA visualization path
        if result.get("ela_visualization"):
            ela_vis_path = result["ela_visualization"]

        if result["success"]:
            pred = result["prediction"]
            conf = result["confidence"]
            icon = "🤖" if "FAKE" in pred else "📷"
            print(f"     {icon}  {pred}  ({conf})")
            if "indicators" in result:
                for ind in result["indicators"]:
                    if "look natural" not in ind and "skipped" not in ind:
                        print(f"         ⚠  {ind}")
            # Method-specific quick stats
            for key, label in [("ela_metrics","ELA"), ("noise_stats","Noise"),
                                ("compression_stats","Comp"), ("texture_stats","Tex"),
                                ("bokeh_stats","Bokeh"), ("boundary_stats","Bound")]:
                if key in result:
                    items = list(result[key].items())[:2]
                    kv    = "  ".join(f"{k}={v}" for k, v in items)
                    print(f"         [{label}] {kv}")
        else:
            print(f"     ⚠️   Failed: {result.get('error','?')[:60]}")
        print()

    # ── Ensemble ──
    ensemble = ensemble_predict(results)

    print("═"*70)
    print("  📊  ENSEMBLE BREAKDOWN:")
    print(f"  {'Method':<44} {'Conf':>6}  {'W→Adj':>7}  {'Contrib':>7}  Dir")
    print(f"  {'─'*44} {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}")
    for m in ensemble.get("method_breakdown", []):
        icon = "🤖" if m["direction"] == "→ FAKE" else "📷"
        note = f"  ◄ {m['note']}" if m.get("note") else ""
        print(f"  {icon} {m['method'][:42]:<42}  {m['confidence']:>6}"
              f"  {m['base_weight']}→{m['adj_weight']:<5}  {m['contribution']:>7.2f}"
              f"  {m['direction']}{note}")

    if ensemble.get("override_triggered"):
        print(f"\n  {ensemble['override_triggered']}")

    raw = ensemble.get("raw_votes", {})
    print()
    print("  🏆  FINAL VERDICT")
    print(f"      {ensemble['final_verdict']}")
    print(f"      Confidence   : {ensemble['confidence']}")
    print(f"      Fake score   : {ensemble['weighted_scores']['fake']}")
    print(f"      Real score   : {ensemble['weighted_scores']['real']}")
    print(f"      Raw votes    : FAKE={raw.get('fake',0)}  REAL={raw.get('real',0)}")
    print(f"      Methods used : {ensemble['methods_used']} / {len(all_methods)}")
    if ensemble['methods_failed'] > 0:
        print(f"      ⚠ Failed     : {ensemble['methods_failed']} method(s)")

    if ela_vis_path:
        print(f"\n  🖼️   ELA Visualization saved → {ela_vis_path}")
        print("       Bright areas = high compression error (REAL regions)")
        print("       Dark/uniform = low error (suspicious AI regions)")
    print("═"*70)

    return ensemble


# ══════════════════════════════════════════════════════════════
# ENTRY POINT  ← Set your image path here ⬇️
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ✅ CHANGE THIS to your image path:
    image_file = "/content/ai-generated-watermark-16x9-grok-elon-musk.jpg"

    # ─────────────────────────────────────────────────────────
    # No need to change anything below this line
    # ─────────────────────────────────────────────────────────
    analyze_image(image_file)