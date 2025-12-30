# Model Training Guide

This guide covers training the face anti-spoofing model from scratch.

## Prerequisites

- GPU with CUDA support (recommended)
- 16GB+ RAM
- Face anti-spoofing dataset (e.g., CelebA-Spoof, OULU-NPU)

## Dataset Preparation

### Supported Datasets

- **CelebA-Spoof**: Recommended (625,537 images)
- **OULU-NPU**: Standard benchmark
- **CASIA-FASD**: Smaller dataset for quick experiments
- **Replay-Attack**: Video-based attacks

### Directory Structure

```
data/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── spoof/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── real/
│   └── spoof/
└── test/
    ├── real/
    └── spoof/
```

### Data Preparation Script

```bash
# Prepare data from raw dataset
python scripts/prepare_data.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --split train:val:test=0.7:0.15:0.15
```

## Training

### Configuration

Training configuration is defined in `src/models/minifas.py`. Key parameters:

```python
class TrainConfig:
    input_size: int = 128          # Input image size
    batch_size: int = 64           # Batch size
    num_epochs: int = 50           # Training epochs
    learning_rate: float = 1e-1    # Initial learning rate
    weight_decay: float = 5e-4     # L2 regularization
    milestones: List[int] = [10, 15, 22, 30]  # LR decay steps
    gamma: float = 0.1             # LR decay factor
```

### Start Training

**Note**: Training script is not yet implemented. You can:

1. **Use the reference implementation** for training
2. **Adapt existing training frameworks** (PyTorch Lightning, etc.)
3. **Implement custom training loop** based on the model architecture

Example training loop structure:

```python
import torch
from src.models.minifas import create_model

# Create model
model = create_model(input_size=128, num_classes=2)
model = model.cuda()

# Optimizer and scheduler
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[10, 15, 22, 30],
    gamma=0.1,
)

# Loss functions
criterion_cls = torch.nn.CrossEntropyLoss()
criterion_ft = torch.nn.MSELoss()  # For Fourier Transform loss

# Training loop
for epoch in range(50):
    model.train()
    
    for batch in train_loader:
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        
        # Forward pass
        cls_output, ft_output = model(images)
        
        # Compute losses
        loss_cls = criterion_cls(cls_output, labels)
        loss_ft = criterion_ft(ft_output, target_ft)  # Need to compute target
        loss = loss_cls + 0.5 * loss_ft
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    model.eval()
    # ... validation code ...
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'checkpoints/epoch_{epoch}.pth')
```

## Export to ONNX

After training, export the best checkpoint to ONNX:

```bash
python scripts/export_onnx.py \
    checkpoints/best_model.pth \
    --output models/best_model.onnx \
    --input-size 128
```

## Quantization

Quantize the model for faster inference and smaller size:

```bash
# Dynamic quantization (no calibration needed)
python scripts/quantize_onnx.py \
    models/best_model.onnx \
    --output models/best_model_quantized.onnx \
    --mode dynamic

# Static quantization (better performance, needs calibration data)
python scripts/quantize_onnx.py \
    models/best_model.onnx \
    --output models/best_model_quantized.onnx \
    --mode static \
    --calibration-dir data/calibration \
    --num-samples 500
```

## Evaluation

Evaluate model performance:

```bash
python scripts/evaluate.py \
    --model models/best_model_quantized.onnx \
    --test-dir data/test \
    --output results/metrics.json
```

Expected metrics:
- **Accuracy**: >97%
- **TPR @ FPR=1%**: >95%
- **AUC**: >0.99
- **Model Size**: <1 MB (quantized)
- **Inference Time**: <10ms (CPU)

## Training Tips

### Data Augmentation

Use aggressive augmentation:
- Random crop and resize
- Color jitter
- Gaussian blur
- Random rotation (±15°)
- Horizontal flip

### Regularization

- Dropout: 0.4 (already in model)
- Weight decay: 5e-4
- Label smoothing: 0.1

### Learning Rate

- Start with 0.1 for batch size 64
- Use cosine annealing or step decay
- Warmup for first 5 epochs

### Mixed Precision

Use mixed precision for faster training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        output = model(images)
        loss = criterion(output, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Advanced Training

### Multi-GPU Training

```python
model = torch.nn.DataParallel(model)
```

### Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed training
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model)
```

### Hyperparameter Tuning

Use Optuna or Ray Tune for automated hyperparameter optimization:

```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    
    # Train model with these hyperparameters
    # Return validation accuracy
    
    return val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## Monitoring

Use TensorBoard for training visualization:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment1')

# Log metrics
writer.add_scalar('Loss/train', loss, epoch)
writer.add_scalar('Accuracy/val', accuracy, epoch)
```

View with:
```bash
tensorboard --logdir runs/
```

## Troubleshooting

### OOM Errors

- Reduce batch size
- Use gradient accumulation
- Enable mixed precision

### Slow Training

- Check data loading (use `num_workers`)
- Profile with PyTorch profiler
- Use GPU monitoring tools

### Poor Convergence

- Check learning rate (too high/low)
- Verify data preprocessing
- Try different optimizers (AdamW, etc.)

## Resources

- [PyTorch Training Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)
- [Model Quantization](https://pytorch.org/docs/stable/quantization.html)

For questions, see [CONTRIBUTING.md](../CONTRIBUTING.md).
