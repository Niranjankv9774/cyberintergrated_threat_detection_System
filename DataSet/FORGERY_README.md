# Multi-Method Image Forgery & Deepfake Detection v6.0

## About
The Image Forgery Detection module has been upgraded to a high-precision **Ensemble System** that combines 19 different forensic, physical, and deep-learning detection methods. This multi-layered approach ensures that even sophisticated AI-generated images (Stable Diffusion, Midjourney, GANs) are correctly identified.

### Detection methods (19 Layers)

The system analyzes images across four primary pillars:

1.  **Forensic Analysis**
    *   **ELA (Error Level Analysis)**: Identifies differences in JPEG compression levels.
    *   **Noise/Sensor Pattern (PRNU)**: Analyzes camera sensor noise consistency.
    *   **EXIF Metadata**: Scans for software signatures (e.g., Photoshop, GIMP).
    *   **Compression Artifacts**: Detects blocking artifacts typical of edited regions.

2.  **Physical & Optical Consistency**
    *   **Lighting & Color Physics**: Checks for non-physical color distributions.
    *   **Shadow & Highlight Physics**: Identifies inconsistent light sources.
    *   **Geometric Distortion**: Scans for warping in fine lines (teeth, eyes).
    *   **Reflection/Refraction**: Analyzes eye reflections and liquid properties.

3.  **Texture & Frequency Analysis**
    *   **Gabor/LBP/Wavelets**: Scans for unnatural smoothness in skin or hair.
    *   **Frequency (FFT+DCT)**: Detects periodic patterns found in GAN-generated images.

4.  **Deep Learning & AI Models**
    *   **HuggingFace AI Detector**: Uses specialized ViT-based AI detection models.
    *   **SDXL/Stable Diffusion Detector**: Tuned for high-resolution diffusion models.
    *   **GAN/Deepfake Detector**: Specifically targets facial swaps and GAN artifacts.
    *   **CLIP Zero-Shot**: Leverages CLIP for semantic forgery detection.

---

## Technical Logic: Weighted Ensemble

The system does not rely on a single model. Instead, it uses a **Weighted Ensemble Predictor**:
- **Weighted Voting**: Reliable methods (like ELA and specialized AI detectors) have higher weights.
- **Outlier Rejection**: If a method's confidence is too low or conflicts wildly with others, it is ignored.
- **Override Rules**: If three high-confidence methods (e.g., SDXL Detector + ELA + Metadata) all agree an image is forged, the system triggers an immediate **Forged** verdict regardless of other scores.

---

## Using the Module

### 1. Uploading Images
1.  Navigate to the **Image Analysis** section in the app.
2.  Upload a photo (works for both portraits and general scenes).
3.  Click **Run Check**.

### 2. Result Interpretation
- **REAL**: The image shows consistent optical, forensic, and geometric properties.
- **Forged**: The image shows signs of AI generation or manual manipulation.

---

## Troubleshooting & Best Practices

- **Lighting**: For the best results, use images with clear, consistent lighting.
- **Resolution**: High-resolution images allow for better texture analysis.
- **Plain Text Storage**: For database stability, the results are stored as simple "Forged" or "Real" labels.

---

## Developer Notes
- **Core Script**: [forgery_detect.py](file:///d:/RISS%202025/HG%20-%20BTech/CyberIntergrated/web/cyberintegratedproject/myapp/forgery_detect.py)
- **Integration**: Linked via `myapp/views.py` using the `analyze_image()` entry point.
- **Dependencies**: Uses PyTorch, Transformers, OpenCV, and SciPy. Ensure `requirements.txt` is installed.
