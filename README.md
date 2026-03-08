# EAST-Caffe: An Efficient and Accurate Scene Text Detector

A Caffe re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2) with **MobileNet V3** backbone, optimized for mobile deployment.

<p align="center">
  <img src="https://github.com/SURFZJY/EAST-caffe/blob/master/results/img_123.jpg" width="420" alt="ICDAR 2013 demo">
  <img src="https://github.com/SURFZJY/EAST-caffe/blob/master/results/img_119.jpg" width="420" alt="ICDAR 2015 demo">
</p>
<p align="center">
  <img src="https://github.com/SURFZJY/EAST-caffe/blob/master/results/a.png" width="420" alt="ID card demo">
</p>

## Features

- **Lightweight backbone**: MobileNet V3 with depthwise separable convolutions and H-Swish activation
- **Dual inference API**: OpenCV DNN (recommended) and native Caffe
- **Mobile deployment**: NCNN / MNN support with INT8 quantization (model size < 2 MB)
- **RBOX output**: Rotated bounding box detection (4 distances + 1 angle)
- **VGG-16 alternative**: Optional VGG backbone for higher accuracy scenarios
- **Cython acceleration**: Optimized geometry map generation for training preprocessing

## Architecture

```
Input (512x512x3)
       |
  MobileNet V3 / VGG-16  (backbone)
       |
  Feature Pyramid (3-level fusion with deconvolution)
       |
  +----+----+
  |         |
Score Map  Geometry Map
(1x128x128) (5x128x128)
  |         |-- 4 channels: distances to top/right/bottom/left edges
  |         |-- 1 channel:  rotation angle
  |
  +----+----+
       |
  RBOX Decode + NMS
       |
  Rotated Bounding Boxes
```

## Table of Contents

- [Installation](#installation)
- [Pretrained Models](#pretrained-models)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Custom Layers](#custom-layers)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation

### Prerequisites

| Dependency | Version | Notes |
|-----------|---------|-------|
| Caffe | > 1.0 | Recommend [SSD branch](https://github.com/weiliu89/caffe/tree/ssd) |
| Python | 2.7 / 3.x | With Caffe Python bindings |
| OpenCV | 3.4.5+ / 4.0+ | Required for DNN inference |
| CUDA | - | GPU training & inference |
| NumPy | - | - |
| Cython | - | For preprocessing acceleration |
| Shapely | - | Polygon operations |

### Setup Steps

1. **Install Caffe** (> 1.0) with Python bindings. The [SSD branch](https://github.com/weiliu89/caffe/tree/ssd) is recommended as it includes ReLU6 support.

2. **Add custom layers** (if not using the SSD branch):
   - **DiceCoefLoss**: Compile from [this repo](https://github.com/HolmesShuan/A-Variation-of-Dice-coefficient-Loss-Caffe-Layer), or use the Python version (`DiceCoefLossLayer` in `pylayerUtils.py` — see commented section in `train.prototxt`).
   - **ReLU6**: Available in [chuanqi305/ssd](https://github.com/chuanqi305/ssd).

3. **Build Cython extension** (accelerates distance calculation during training):

   ```bash
   cd geo_map_cython_lib
   sh build_ext.sh
   ```

## Pretrained Models

| Dataset | Download | Extract Code |
|---------|----------|-------------|
| ICDAR 2013 (train) | [Baidu Pan](https://pan.baidu.com/s/1_daEvvt7ur3FdXVxVKSF9A) | `krdu` |
| ICDAR 2015 (train) | [Baidu Pan](https://pan.baidu.com/s/1DLTJDiRIqihE6ad5uiHEFA) | `pn0w` |
| Fake ID Card (single char) | [Baidu Pan](https://pan.baidu.com/s/1KpG7xFPChyJMftAGR2SdYw) | `m70q` |

Place downloaded `.caffemodel` files in the `snapshot/` directory.

## Training

### Dataset Preparation

Organize your dataset as follows:

```
dataset_dir/
├── train_images/    # Training images
├── train_gts/       # Ground truth annotation files
├── test_images/     # Test images
└── test_gts/        # Test annotations
```

**Annotation format** (one line per text instance):

```
x1,y1,x2,y2,x3,y3,x4,y4,transcription
```

where `(x1,y1)...(x4,y4)` are the four corners of the text quadrilateral.

### Configure Dataset Path

Edit `pylayerUtils.py` and update the `datasetDict` in the `DataLayer.setup()` method to point to your dataset directory:

```python
datasetDict = {
    'my_dataset': '/path/to/your/dataset',
}
self.dataset = 'my_dataset'
```

### Start Training

```bash
# Single GPU
python train.py --gpu 0 --initmodel pretrained.caffemodel

# Multi-GPU (pass GPU IDs)
python train.py --gpu 0,1,2,3 --initmodel pretrained.caffemodel
```

### Solver Configuration

The default `solver.prototxt` uses:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 (fixed) |
| Max iterations | 200,000 |
| Snapshot interval | 200 iterations |
| Batch momentum | 0.9 / 0.999 |

## Inference

### Single Image

```bash
# OpenCV DNN inference (recommended, no Caffe dependency at runtime)
python demo.py --input imgs/img_123.jpg \
               --model_def models/mbv3/deploy.prototxt \
               --model_weights snapshot/ic15_iter_32000.caffemodel \
               --infer dnn

# Native Caffe inference
python demo.py --input imgs/img_123.jpg \
               --model_def models/mbv3/deploy.prototxt \
               --model_weights snapshot/ic15_iter_32000.caffemodel \
               --infer caffe --gpu 0
```

### Batch Processing

Modify `demo.py` to call `batch_demo(input_dir, output_dir)` instead of `single_demo(...)`:

```bash
python demo.py --input_dir imgs/ic15_test \
               --output_dir results \
               --model_def models/mbv3/deploy.prototxt \
               --model_weights snapshot/ic15_iter_32000.caffemodel
```

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `imgs/ic15_train/img_896.jpg` | Path to a single input image |
| `--input_dir` | `imgs/ic15_test` | Directory of images for batch processing |
| `--output_dir` | `results` | Output directory for results |
| `--model_def` | `models/mbv3/deploy.prototxt` | Network definition (prototxt) |
| `--model_weights` | `snapshot/ic15_iter_32000.caffemodel` | Trained model weights |
| `--thr` | `0.95` | Confidence threshold |
| `--nms` | `0.1` | NMS threshold |
| `--infer` | `dnn` | Inference backend: `dnn` or `caffe` |
| `--gpu` | `5` | GPU device ID (Caffe backend only) |

## Project Structure

```
EAST-caffe/
├── train.py                 # Training entry point
├── demo.py                  # Inference / demo script
├── icdar.py                 # Data loading, augmentation, geometry map generation
├── pylayerUtils.py          # Custom Caffe Python layers (DataLayer, loss layers)
├── solver.prototxt          # Caffe solver configuration
├── models/
│   ├── mbv3/
│   │   ├── train.prototxt   # MobileNet V3 training network
│   │   └── deploy.prototxt  # MobileNet V3 deploy network
│   └── vgg/
│       └── train.prototxt   # VGG-16 training network
├── geo_map_cython_lib/
│   ├── gen_geo_map.pyx      # Cython-optimized distance calculation
│   ├── setup.py             # Cython build config
│   └── build_ext.sh         # Build script
├── imgs/                    # Sample test images
├── results/                 # Inference output directory
└── snapshot/                # Model checkpoints (user-created)
```

## Custom Layers

### DataLayer (`pylayerUtils.py`)

Online data loading with random scaling (0.5x–3.0x), text-aware cropping, and geometry map generation via Cython.

**Outputs**: `data` (image), `score_map` (1ch), `geo_map` (5ch)

### DiceCoefLossLayer (`pylayerUtils.py`)

Dice coefficient loss for score map segmentation:

```
Loss = 1 - (2 * intersection + 1) / (sum_pred + sum_gt + 1)
```

### RBoxLossLayer (`pylayerUtils.py`)

Combined geometry loss with AABB IoU and angle components:

```
L_AABB  = -log((area_intersect + 1) / (area_union + 1))
L_theta = 1 - cos(theta_pred - theta_gt)
L_total = mean((L_AABB + 20 * L_theta) * score_gt)
```

## Acknowledgements

This project is inspired by and builds upon:

- [argman/EAST](https://github.com/argman/EAST) — Original TensorFlow implementation
- [YukangWang/TextField](https://github.com/YukangWang/TextField)
- [chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)

## License

[MIT License](LICENSE) &copy; 2019 SURFZJY
