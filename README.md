# ArUco Barcode Object Detection Training Project

A comprehensive Python project for training and deploying object detection models to detect ArUco barcodes using deep learning.

## Project Goals

Train a robust object detection model capable of:
- Detecting ArUco barcodes in images and video streams
- Localizing barcode positions with bounding boxes
- Handling various lighting conditions, scales, and orientations
- Real-time inference on embedded devices

## Project Structure

```
capstone/
├── .github/
│   └── copilot-instructions.md      # Development guidelines
├── src/
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── aruco_generator.py       # ArUco barcode generation
│   │   ├── dataset_utils.py         # Dataset utilities
│   │   └── generate_dataset.py      # Main dataset generation script
│   ├── models/
│   │   ├── __init__.py
│   │   └── detector.py              # Detection model wrapper
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training logic
│   │   └── train.py                 # Training entry point
│   └── inference/
│       ├── __init__.py
│       ├── detector.py              # Inference utilities
│       └── detect.py                # Detection entry point
├── data/                            # Dataset directory
├── runs/                            # Model checkpoints and results
├── requirements.txt                 # Python dependencies
├── setup.py                         # Project setup
└── README.md                        # This file
```

## Features

### Dataset Generation
- Generates synthetic ArUco barcode images
- Supports various barcode sizes and positions
- Applies realistic augmentations (rotation, lighting, noise)
- Automatic YOLO format annotation generation

### Model Training
- YOLOv5 based object detection
- Support for custom training parameters
- Checkpoint saving and resume capability
- Training metrics visualization

### Inference
- Real-time barcode detection
- Image and video stream support
- Confidence score filtering
- XML/JSON export capabilities

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB+ recommended for training)
- GPU recommended (NVIDIA with CUDA support)

## Installation

1. Clone or navigate to the project directory
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Generate Dataset

```bash
python src/dataset/generate_dataset.py \
    --output data/synthetic_aruco \
    --num-images 1000 \
    --img-size 640
```

### Train Model

```bash
python src/training/train.py \
    --data-path data/synthetic_aruco \
    --epochs 50 \
    --batch-size 16 \
    --img-size 640
```

### Run Inference

```bash
python src/inference/detect.py \
    --image test_image.jpg \
    --weights runs/detect/weights/best.pt \
    --conf-threshold 0.5
```

## Configuration

Edit configuration parameters in each script or pass them as command-line arguments. Key parameters:

- `img-size`: Input image size (default: 640)
- `batch-size`: Training batch size (default: 16)
- `epochs`: Number of training epochs (default: 50)
- `learning-rate`: Initial learning rate (default: 0.001)
- `num-images`: Dataset size (default: 1000)

## Model Training

The project uses YOLOv5 for object detection. Training includes:

1. Data augmentation (rotation, scaling, color jitter)
2. Multi-scale training
3. Learning rate scheduling
4. Model checkpointing
5. Evaluation metrics computation

## Evaluation

After training, evaluate the model:

```python
from src.training.trainer import Trainer

trainer = Trainer()
metrics = trainer.evaluate('path/to/test/data')
print(f"mAP50: {metrics['mAP50']}")
```

## Results

Training results are saved in the `runs/` directory with:
- `weights/best.pt` - Best model checkpoint
- `results.csv` - Training metrics per epoch
- `confusion_matrix.png` - Confusion matrix
- `results.png` - Performance plots

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Reduce image size
- Use mixed precision training

### Slow Training
- Reduce dataset size initially
- Use data loader num_workers
- Enable GPU acceleration

### Poor Detection Results
- Increase training epochs
- Generate more diverse dataset
- Adjust data augmentation parameters

## Contributing

When modifying the code:
1. Follow PEP 8 style guidelines
2. Add docstrings to functions
3. Update README if adding features
4. Test changes with small dataset first

## Future Enhancements

- [ ] Support for other barcode formats (QR codes, Data Matrix)
- [ ] Mobile deployment (ONNX, TensorFlow Lite)
- [ ] Web interface for testing
- [ ] Real-time video stream processing
- [ ] Multi-GPU training support
- [ ] Quantization and pruning for inference optimization

## License

This project is provided as-is for educational purposes.

## References

- [OpenCV ArUco Documentation](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
