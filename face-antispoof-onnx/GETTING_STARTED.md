# Getting Started with Face Liveness ONNX

This guide will help you set up and run the face anti-spoofing system.

## Prerequisites

- Python 3.8 - 3.11
- pip
- (Optional) CUDA-capable GPU for training

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd face-antispoof-onnx
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# For inference only (minimal)
pip install numpy opencv-python onnxruntime pillow

# For full development (training, testing, etc.)
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Download Models

You need to obtain the pre-trained ONNX models. You have several options:

#### Option A: Copy from Reference Implementation

If you have access to the reference implementation:

```bash
# Create models directory
mkdir models

# Copy model files
cp path/to/ref/models/best_model_quantized.onnx models/
cp path/to/ref/models/detector_quantized.onnx models/
```

#### Option B: Download from GitHub Releases

Download from the releases page or provided URL and place in `models/` directory:
- `best_model_quantized.onnx` (600 KB)
- `detector_quantized.onnx` (~2 MB)

#### Option C: Train Your Own

See [Training Guide](docs/TRAINING.md) for instructions.

#### Verify Installation

```bash
python scripts/download_models.py --verify
```

Expected output:
```
✓ best_model_quantized.onnx (0.60 MB)
✓ detector_quantized.onnx (2.00 MB)
✓ All required models found!
```

## Quick Test

### Test with Demo

```bash
# Image test
python demo.py --mode image --source path/to/image.jpg

# Webcam test
python demo.py --mode webcam

# Video test
python demo.py --mode video --source path/to/video.mp4
```

### Programmatic Usage

```python
from src.inference.inference import FaceAntiSpoof

# Initialize
detector = FaceAntiSpoof(
    model_path="models/best_model_quantized.onnx",
    detector_path="models/detector_quantized.onnx",
)

# Predict on image
import cv2
image = cv2.imread("test.jpg")
result = detector.predict(image)

print(f"Label: {result['label']}")  # 'real' or 'spoof'
print(f"Confidence: {result['confidence']:.2%}")
```

## Next Steps

- **For Users**: See [README.md](README.md) for API documentation
- **For Developers**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
- **For Training**: See [docs/TRAINING.md](docs/TRAINING.md) for training instructions
- **For Deployment**: See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for production setup

## Common Issues

### Missing Models Error

```
FileNotFoundError: Model not found: models/best_model_quantized.onnx
```

**Solution**: Follow step 4 above to download/copy models.

### Import Errors

```
ModuleNotFoundError: No module named 'onnxruntime'
```

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Low FPS on CPU

The quantized models are optimized for CPU inference. Expected performance:
- CPU: ~60-100 FPS (quantized model)
- GPU: ~200-400 FPS

For GPU acceleration:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### Camera Not Found

```
Error: Could not open camera
```

**Solution**: 
- Check camera permissions
- Try different camera index: `--camera-id 1`
- On Linux: Add user to `video` group

## Support

For issues and questions:
1. Check [docs/LIMITATIONS.md](docs/LIMITATIONS.md)
2. Search existing issues
3. Create a new issue with details

## License

See [LICENSE](LICENSE) for details.
