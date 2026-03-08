# CLAUDE.md — Development Guide for EAST-Caffe

## Project Overview

EAST-Caffe is a Caffe-based implementation of the EAST (Efficient and Accurate Scene Text) detector. It uses MobileNet V3 as the primary backbone for lightweight text detection, with an optional VGG-16 backbone. The project outputs rotated bounding boxes (RBOX) for detected text regions.

## Quick Reference

```bash
# Build Cython extension (required before training)
cd geo_map_cython_lib && sh build_ext.sh && cd ..

# Train (single GPU)
python train.py --gpu 0 --initmodel snapshot/pretrained.caffemodel

# Inference with OpenCV DNN
python demo.py --input imgs/img_123.jpg --model_def models/mbv3/deploy.prototxt --model_weights snapshot/model.caffemodel --infer dnn

# Inference with Caffe
python demo.py --input imgs/img_123.jpg --infer caffe --gpu 0
```

## Repository Layout

| Path | Purpose |
|------|---------|
| `train.py` | Training entry point — wraps Caffe AdamSolver |
| `demo.py` | Inference script (single image or batch) |
| `icdar.py` | Data loading, augmentation, and geometry map generation |
| `pylayerUtils.py` | Custom Python layers: `DataLayer`, `DiceCoefLossLayer`, `RBoxLossLayer` |
| `solver.prototxt` | Caffe solver config (Adam, lr=0.001 with step decay, 200K iterations) |
| `models/mbv3/train.prototxt` | MobileNet V3 training network definition |
| `models/mbv3/deploy.prototxt` | MobileNet V3 deployment network definition |
| `models/vgg/train.prototxt` | VGG-16 training network definition |
| `geo_map_cython_lib/` | Cython module for fast distance calculation |
| `imgs/` | Sample test images |
| `results/` | Inference output directory |
| `snapshot/` | Model checkpoint directory (user-created) |

## Architecture

- **Input**: 512x512x3 BGR images (mean-subtracted: [103.94, 116.78, 123.68])
- **Backbone**: MobileNet V3 (lightweight) or VGG-16
- **Decoder**: 3-level feature pyramid with deconvolution-based upsampling
- **Output maps** (at 1/4 resolution = 128x128):
  - Score map: 1 channel — text region probability (sigmoid)
  - Geometry map: 4 channels — distances to bounding box edges (sigmoid * 512)
  - Angle map: 1 channel — rotation angle [-pi/2, pi/2]
- **Post-processing**: RBOX decode → NMS via `cv2.dnn.NMSBoxesRotated`

## Key Dependencies

- **Caffe** > 1.0 with Python bindings (recommend [SSD branch](https://github.com/weiliu89/caffe/tree/ssd))
- **OpenCV** 3.4.5+ or 4.0+ (for DNN inference and NMS)
- **Python packages**: numpy, scipy, cython, shapely, tqdm, cv2
- **Custom C++ layers** (compile into Caffe or use Python alternatives):
  - `DiceCoefLoss` — [source](https://github.com/HolmesShuan/A-Variation-of-Dice-coefficient-Loss-Caffe-Layer)
  - `ReLU6` — available in [chuanqi305/ssd](https://github.com/chuanqi305/ssd)

## Loss Functions

The training loss has two components:

1. **Score loss** (DiceCoefLoss): `1 - Dice(pred, gt)` with loss weight `0.01`
2. **Geometry loss** (RBoxLoss) with loss weight `1.0`:
   - AABB IoU: `-log((area_intersect + 1) / (area_union + 1))`
   - Angle: `1 - cos(theta_pred - theta_gt)` weighted by factor `20`
   - Combined: `mean((L_AABB + 20 * L_theta) * score_gt)`

## Dataset Format

```
dataset_dir/
├── train_images/    # .jpg/.png images
├── train_gts/       # Annotation text files (same basename as images)
├── test_images/
└── test_gts/
```

**Annotation format** (per line): `x1,y1,x2,y2,x3,y3,x4,y4,transcription`

To add a new dataset: edit `pylayerUtils.py` → `DataLayer.setup()` → update `datasetDict` and `self.dataset`.

## Development Notes

### Modifying the Network

- Training network definitions are in `models/mbv3/train.prototxt` and `models/vgg/train.prototxt`
- Deployment network (no data/loss layers) is `models/mbv3/deploy.prototxt`
- When changing layer structure, keep both train and deploy prototxt files in sync
- The Python data layer (`DataLayer` in `pylayerUtils.py`) is referenced by `train.prototxt` — changes to its interface require updating both files

### Adding a New Backbone

1. Create a new directory under `models/` (e.g., `models/resnet/`)
2. Define `train.prototxt` with the data layer, backbone, feature pyramid decoder, and loss layers
3. Define `deploy.prototxt` with only the backbone and decoder (no data/loss layers)
4. Update `solver.prototxt` to point to the new `train.prototxt`

### Data Augmentation

Augmentation is handled in `icdar.py` within the `generator()` function:
- Random scaling: 0.5x to 3.0x
- Random cropping with text-aware region selection
- Background region sampling (when no text is present in crop)
- Geometry map generation uses Cython (`geo_map_cython_lib`) for performance

### Common Issues

- **DiceCoefLoss not found**: Either compile the C++ layer into Caffe, or uncomment the Python `DiceCoefLossLayer` in `train.prototxt`
- **ReLU6 not found**: Use the SSD branch of Caffe which includes this layer
- **Cython build fails**: Ensure `cython` and a C compiler are installed; check `geo_map_cython_lib/setup.py`
- **Image size must be multiple of 32**: The `resize_image()` function in `demo.py` handles this automatically

### Inference Backends

- **`dnn` (default)**: Uses `cv2.dnn.readNet()` — no Caffe installation needed at inference time, portable
- **`caffe`**: Uses native Caffe Python API — requires full Caffe installation, supports GPU selection

### Output Format

Inference outputs per-image text files in `results/` with format:
```
x1,y1,x2,y2,x3,y3,x4,y4, RESULT
```
Each line represents one detected text region as a rotated quadrilateral.
