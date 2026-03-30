# ANCHURUS - Agile Network for Complex High-speed Unordered Reconstruction of Unitary Spaces

A lightweight, efficient depth estimation network designed for monocular depth prediction using a student-teacher learning framework.

## Overview

**ANCHURUS** is a neural network architecture for single-image depth estimation. It uses a **MobileNetV2-based encoder-decoder architecture** combined with adaptive learning strategies to achieve high-quality depth maps from RGB images.

### Name Origin

The name **ANCHURUS** draws inspiration from classical Greek mythology. In the student-teacher learning framework:
- **MiDaS** (the teacher model) is named after King Midas, known for his ability to perceive and transform matter
- **ANCHURUS** (the student model) represents the son of Midas in mythology, symbolizing knowledge passed from teacher to student through distillation

This mythological connection reflects the core principle of the architecture: a lightweight student network inherits the depth-perception capabilities from a more powerful teacher model, just as a son inherits wisdom from his father.

### Key Features

- 🚀 **Lightweight**: Built on MobileNetV2 for efficient inference
- 📊 **Student-Teacher Framework**: Leverages knowledge distillation for improved performance
- 🎯 **Adaptive Learning**: Dynamic learning rate adjustment based on training progress
- 🔄 **Multiple Model Versions**: 5 pre-trained model versions available (v.1 - v.5)
- ⚡ **CUDA Support**: GPU acceleration for faster training and inference

## Project Structure

```
Depth_estimation/
├── Training/
│   ├── downloader.py          # Data downloading utility
│   ├── student_trainer.py     # Main training script with adaptive learning
│   └── teacher_pred.py        # Teacher model predictions
├── Using/
│   ├── Depth_estimation.py    # Inference script
│   └── models/
│       ├── ANCHURUS(v.1).pth
│       ├── ANCHURUS(v.2).pth
│       ├── ANCHURUS(v.3).pth
│       ├── ANCHURUS(v.4).pth
│       └── ANCHURUS(v.5).pth
└── README.md
```

## Architecture

### Model Components

| Component | Details |
|-----------|---------|
| **Encoder** | MobileNetV2 pre-trained backbone (ImageNet) |
| **Decoder** | 5 upsampling blocks with progressive refinement |
| **Output** | Single-channel depth map (normalized 0-1) via Sigmoid |
| **Input Size** | 256×256 RGB images |

### Model Pipeline

```
RGB Image (256×256) 
    ↓
MobileNetV2 Encoder (extracts features)
    ↓
Decoder (5 upsampling blocks)
    ↓
Depth Map (256×256, 0-1 normalized)
```

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher (optional, for GPU acceleration)
- **cuDNN**: 8.0 or higher (optional, required if using CUDA)
- **pip**: Latest version recommended
- **Virtual Environment**: Recommended (venv, conda, or poetry)

### Quick Setup

1. **Clone or download the project**:
```bash
cd Depth_estimation
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | ≥2.0.0 | Deep learning framework |
| **torchvision** | ≥0.15.0 | Computer vision utilities and pre-trained models |
| **opencv-python** | ≥4.8.0 | Image/video processing |
| **numpy** | ≥1.24.0 | Numerical computing |
| **tqdm** | ≥4.65.0 | Progress bars |
| **yt-dlp** | ≥2023.12.0 | YouTube video downloading (training only) |

### Optional Dependencies

- **CUDA Toolkit**: For GPU acceleration (install from [NVIDIA](https://developer.nvidia.com/cuda-toolkit))
- **cuDNN**: CUDA Deep Neural Network library (install from [NVIDIA](https://developer.nvidia.com/cudnn))
- **tensorboard**: For training visualization (`pip install tensorboard`)

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

## Usage

### Inference

Run depth estimation on images or videos:

```python
from Using.Depth_estimation import *

# Load model (v.5 recommended)
model = ANCHURUS().to(DEVICE)
model.load_state_dict(torch.load("models/ANCHURUS(v.5).pth", map_location=DEVICE))
model.eval()

# Process image
depth_map = estimate_depth(image_path)
```

### Training

Train a new model with the student-teacher framework:

```bash
python Training/student_trainer.py
```

#### Training Configuration

Edit `student_trainer.py` to adjust:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `BASE_DIR` | `processed_data` | Dataset directory path |
| `IMG_SIZE` | `(256, 256)` | Input image resolution |
| `BATCH_SIZE` | `32` | Batch size for training |
| `BASE_LR` | `1e-4` | Initial learning rate |
| `KICK_LR` | `1e-3` | Impulse learning rate for recovery |
| `MIN_PROGRESS` | `0.0001` | Minimum accuracy improvement threshold |
| `PATIENCE` | `4` | Epochs to tolerate slow progress |
| `MAX_KICKS` | `3` | Maximum learning rate kicks |

### Data Preparation

Data should be organized as:

```
processed_data/
├── rgb/
│   ├── video_1/
│   │   ├── frame_0.jpg
│   │   └── ...
│   └── ...
└── depth/
    ├── video_1/
    │   ├── frame_0.png
    │   └── ...
    └── ...
```

## Model Versions

| Version | Training Approach | Use Case |
|---------|-------------------|----------|
| **v.1** | Initial baseline | Reference |
| **v.2** | Improved hyperparameters | Better generalization |
| **v.3** | Extended training | Enhanced accuracy |
| **v.4** | Adaptive learning | Robust performance |
| **v.5** | Full optimization | **Recommended** |

## Performance Considerations

### Memory Requirements

- **Inference**: ~150 MB GPU memory
- **Training**: ~4-6 GB GPU memory (batch size 32)

### Speed

- **Inference**: ~30-50 fps on modern GPUs (RTX 2080+)
- **Training**: ~2-4 seconds per 256×256 batch

## Features Highlight

### Adaptive Learning Strategy

The trainer implements a **staircase learning rate adjustment**:

1. **Normal Progress**: Train with `BASE_LR`
2. **Stagnation Detection**: If accuracy improvement < `MIN_PROGRESS` for `PATIENCE` epochs
3. **Recovery Kick**: Increase learning rate to `KICK_LR` for restart
4. **Iteration**: Up to `MAX_KICKS` attempts before stopping

### Student-Teacher Framework

- **Teacher Model**: Provides supervision signal
- **Student Model**: Learns efficient representations
- **Knowledge Distillation**: Transfers teacher knowledge to lightweight student

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `BATCH_SIZE` or use `DEVICE = "cpu"` |
| Model not found | Verify `MODEL_PATH` points to correct `.pth` file |
| Low accuracy | Try a newer model version (v.5) or retrain with more data |
| Slow training | Enable CUDA acceleration or increase `BATCH_SIZE` |

## References

- **MobileNetV2**: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue or contact the project maintainers.

---

**Last Updated**: March 2026
