# Face Liveness ONNX

Lightweight, production-ready face liveness (anti-spoof) pipeline built on MiniFAS with ONNX Runtime. Ships a 600 KB quantized model, fast inference, and a simple demo.

## What you get
- Small ONNX models (quantized and FP32) with fast CPU/GPU inference
- Face detector + liveness model with temporal smoothing for video
- Quality checks, confidence scores, and ready-to-use OpenCV demo
- Export, quantize, and benchmarking scripts

## Install
```bash
git clone https://github.com/yourusername/face-liveness-onnx.git
cd face-liveness-onnx
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
GPU option: `pip install onnxruntime-gpu` (uninstall `onnxruntime` first if needed).

## Quick start
- Demo webcam: `python demo.py --camera 0`
- Demo image: `python demo.py --image path/to/face.jpg --visualize`

Programmatic:
```python
from src.inference import FaceAntiSpoof
from src.detection import FaceDetector
import cv2

detector = FaceDetector("models/detector_quantized.onnx")
model = FaceAntiSpoof("models/best_model_quantized.onnx", threshold=0.5)
frame = cv2.imread("test.jpg")
for det in detector.detect(frame):
    crop = detector.crop_face(frame, det, padding=1.5)
    print(model.predict(crop))
```

## Models
Required: `models/detector_quantized.onnx`, `models/best_model_quantized.onnx`

If missing, run `python scripts/download_models.py --verify` for instructions.

## Docs
- Quickstart: `GETTING_STARTED.md`
- Deployment/API: `docs/DEPLOYMENT.md`
- Limitations and edge cases: `docs/LIMITATIONS.md`

## Contributing
See `CONTRIBUTING.md` for a short checklist (setup, lint, test, PR). Issues and PRs welcome.

## License
Apache 2.0
