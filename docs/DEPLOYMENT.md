# Deployment Guide

This guide covers deploying the face anti-spoofing system in production.

## Production Checklist

- [ ] Model files available and verified
- [ ] Hardware requirements met
- [ ] Security considerations addressed
- [ ] Monitoring and logging configured
- [ ] Error handling tested
- [ ] Performance benchmarked

## Deployment Options

### 1. REST API (Flask/FastAPI)

#### Using FastAPI

```python
# api.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from src.inference.inference import FaceAntiSpoof

app = FastAPI(title="Face Liveness API")

# Initialize detector (singleton)
detector = FaceAntiSpoof(
    model_path="models/best_model_quantized.onnx",
    detector_path="models/detector_quantized.onnx",
    use_gpu=False,
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict if face is real or spoof."""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Predict
        result = detector.predict(image)
        
        return JSONResponse({
            "success": True,
            "label": result["label"],
            "confidence": float(result["confidence"]),
            "bbox": result.get("bbox", None),
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
```

Run with:
```bash
pip install fastapi uvicorn python-multipart
uvicorn api:app --host 0.0.0.0 --port 8000
```

#### Using Flask

```python
# api_flask.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
from src.inference.inference import FaceAntiSpoof

app = Flask(__name__)
detector = FaceAntiSpoof(
    model_path="models/best_model_quantized.onnx",
    detector_path="models/detector_quantized.onnx",
)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    
    # Read image
    nparr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Predict
    result = detector.predict(image)
    
    return jsonify({
        "label": result["label"],
        "confidence": float(result["confidence"]),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
```

### 2. Docker Container

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn python-multipart

# Copy application
COPY . .

# Download models (or copy from build context)
# COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t face-antispoof-api .
docker run -p 8000:8000 face-antispoof-api
```

### 3. Edge Deployment

For edge devices (Raspberry Pi, Jetson Nano):

```python
# edge_detector.py
import cv2
from src.inference.inference import VideoAntiSpoof

# Use quantized model for better performance
detector = VideoAntiSpoof(
    model_path="models/best_model_quantized.onnx",
    detector_path="models/detector_quantized.onnx",
    use_gpu=False,  # Most edge devices use CPU
    temporal_window=3,  # Smaller window for faster response
)

# Process camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    result = detector.process_frame(frame)
    
    # Display or send to cloud
    print(f"Label: {result['label']} ({result['confidence']:.2%})")
    
    # Trigger action if spoof detected
    if result['label'] == 'spoof' and result['confidence'] > 0.9:
        trigger_alert()
```

### 4. Mobile Deployment

For mobile apps, export to TensorFlow Lite or ONNX Mobile:

```python
# Convert to TensorFlow Lite
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# Load ONNX model
onnx_model = onnx.load("models/best_model_quantized.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("models/tf_model")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("models/tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)
```

## Performance Optimization

### CPU Optimization

```python
import onnxruntime as ort

# Enable optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4  # Adjust based on CPU cores

session = ort.InferenceSession(
    "models/best_model_quantized.onnx",
    sess_options=sess_options,
)
```

### GPU Optimization

```python
# Use GPU provider
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
    }),
    'CPUExecutionProvider',
]

session = ort.InferenceSession(
    "models/best_model_quantized.onnx",
    providers=providers,
)
```

### Batching

Process multiple images in parallel:

```python
from src.inference.inference import FaceAntiSpoof

detector = FaceAntiSpoof(...)

# Process batch
images = [image1, image2, image3]
results = [detector.predict(img) for img in images]

# Or implement batch prediction
def predict_batch(images: List[np.ndarray]) -> List[dict]:
    # Detect faces in all images
    faces = [detector.detector.detect(img) for img in images]
    
    # Batch preprocess
    # ... batch inference ...
    
    return results
```

## Monitoring

### Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler(
    "logs/antispoof.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[handler]
)

logger = logging.getLogger(__name__)

# Log predictions
logger.info(f"Prediction: {result['label']} ({result['confidence']:.2%})")
```

### Metrics

Track key metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
predictions_total = Counter(
    'antispoof_predictions_total',
    'Total number of predictions',
    ['label']
)

prediction_duration = Histogram(
    'antispoof_prediction_duration_seconds',
    'Time spent processing prediction',
)

confidence_gauge = Gauge(
    'antispoof_confidence',
    'Prediction confidence',
)

# Use in code
with prediction_duration.time():
    result = detector.predict(image)

predictions_total.labels(label=result['label']).inc()
confidence_gauge.set(result['confidence'])
```

## Security Considerations

### 1. Input Validation

```python
def validate_image(image: np.ndarray) -> bool:
    """Validate image meets requirements."""
    # Check dimensions
    if image.shape[0] < 64 or image.shape[1] < 64:
        raise ValueError("Image too small")
    
    if image.shape[0] > 4096 or image.shape[1] > 4096:
        raise ValueError("Image too large")
    
    # Check channels
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image must be RGB")
    
    return True
```

### 2. Rate Limiting

```python
from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")  # 10 requests per minute
async def predict(request: Request, file: UploadFile):
    # ... prediction code ...
    pass
```

### 3. Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict")
async def predict(file: UploadFile, api_key: str = Security(verify_api_key)):
    # ... prediction code ...
    pass
```

## Scaling

### Load Balancing

Use NGINX for load balancing:

```nginx
# nginx.conf
upstream antispoof_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://antispoof_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-antispoof
spec:
  replicas: 3
  selector:
    matchLabels:
      app: face-antispoof
  template:
    metadata:
      labels:
        app: face-antispoof
    spec:
      containers:
      - name: api
        image: face-antispoof-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

## Best Practices

1. **Model Versioning**: Track model versions and support A/B testing
2. **Graceful Degradation**: Handle errors gracefully, return meaningful messages
3. **Caching**: Cache face detection results for video streams
4. **Async Processing**: Use async/await for I/O operations
5. **Health Checks**: Implement /health endpoint for monitoring
6. **Documentation**: Provide OpenAPI/Swagger docs
7. **Testing**: Load test with realistic traffic patterns

## Resources

- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/deployment/)
- [Docker Security](https://docs.docker.com/engine/security/)

For questions, see [CONTRIBUTING.md](../CONTRIBUTING.md).
