# Gate Marker Setup Guide

This guide explains how to import your 4 prototype gate corner markers into the training dataset.

## Your Marker Classes

The system now supports 4 distinct marker classes for gate corners:
- **Class 0**: `top-left`
- **Class 1**: `top-right`
- **Class 2**: `bottom-left`
- **Class 3**: `bottom-right`

## Setup Steps

### Step 1: Organize Your Marker Files

Create a directory for your markers:
```
data/
  └── gate_markers/
      ├── marker_1.png
      ├── marker_2.png
      ├── marker_3.png
      └── marker_4.png
```

### Step 2: Create Marker Mapping (Option A - Explicit Mapping)

Edit `data/marker_config_template.json` with your filenames:

```json
{
  "markers": {
    "your_topleft_marker.png": "top-left",
    "your_topright_marker.png": "top-right",
    "your_bottomleft_marker.png": "bottom-left",
    "your_bottomright_marker.png": "bottom-right"
  }
}
```

Save as `data/marker_config.json`

### Step 3: Import Markers

**Option A - Using explicit mapping file:**
```bash
python src/dataset/import_prototypes.py \
  --prototypes data/gate_markers \
  --dataset data/aruco_dataset \
  --mapping data/marker_config.json \
  --prefix gate
```

**Option B - Auto-detect from filenames:**
If your files are named like `top-left.png`, `topright.png`, `tl.png`, etc., use:
```bash
python src/dataset/import_prototypes.py \
  --prototypes data/gate_markers \
  --dataset data/aruco_dataset \
  --auto-detect \
  --prefix gate
```

**Option C - From command line:**
```bash
python src/dataset/import_prototypes.py \
  --prototypes data/gate_markers \
  --dataset data/aruco_dataset \
  --mapping data/marker_config.json
```

## Examples

### Example 1: Simple filenames with auto-detection
```
gate_markers/
  ├── top-left.png
  ├── top-right.png
  ├── bottom-left.png
  └── bottom-right.png

# Command:
python src/dataset/import_prototypes.py --prototypes data/gate_markers --auto-detect
```

### Example 2: Custom filenames with mapping
```
gate_markers/
  ├── frame_01.png  # → top-left
  ├── frame_02.png  # → top-right
  ├── frame_03.png  # → bottom-left
  └── frame_04.png  # → bottom-right

# marker_config.json:
{
  "markers": {
    "frame_01.png": "top-left",
    "frame_02.png": "top-right",
    "frame_03.png": "bottom-left",
    "frame_04.png": "bottom-right"
  }
}

# Command:
python src/dataset/import_prototypes.py --prototypes data/gate_markers --mapping data/marker_config.json
```

### Example 3: With annotations (if you have bounding box labels)
```bash
python src/dataset/import_prototypes.py \
  --prototypes data/gate_markers \
  --annotations data/gate_markers/annotations \
  --mapping data/marker_config.json
```

If annotations don't exist, the system will create full-image annotations (marker covers entire image).

## Combining with Synthetic Data

To use both synthetic and real markers:

```bash
# 1. Generate synthetic dataset
python src/dataset/generate_dataset.py --output data/aruco_dataset --num-images 500 --corner-classes

# 2. Add your real gate markers
python src/dataset/import_prototypes.py \
  --prototypes data/gate_markers \
  --dataset data/aruco_dataset \
  --mapping data/marker_config.json
```

## Dataset Structure After Import

```
data/aruco_dataset/
├── images/
│   ├── train/
│   │   ├── aruco_000000.jpg
│   │   ├── ...
│   │   ├── gate_top_left_000.png
│   │   ├── gate_top_right_000.png
│   │   ├── gate_bottom_left_000.png
│   │   └── gate_bottom_right_000.png
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   │   ├── aruco_000000.txt
│   │   ├── gate_top_left_000.txt (class_id=0)
│   │   ├── gate_top_right_000.txt (class_id=1)
│   │   ├── gate_bottom_left_000.txt (class_id=2)
│   │   └── gate_bottom_right_000.txt (class_id=3)
│   ├── val/
│   └── test/
└── dataset.yaml
```

## Annotation Format

Each `.txt` file contains YOLO format annotations:
```
class_id x_center y_center width height
```

Example for top-left marker covering entire image:
```
0 0.5 0.5 1.0 1.0
```

If you have multiple markers in an image:
```
0 0.25 0.25 0.2 0.2   # top-left marker
1 0.75 0.25 0.2 0.2   # top-right marker
2 0.25 0.75 0.2 0.2   # bottom-left marker
3 0.75 0.75 0.2 0.2   # bottom-right marker
```

## Troubleshooting

**Error: "Could not determine corner for..."**
- Make sure your filenames match the auto-detect patterns OR
- Provide an explicit mapping in a JSON file OR
- Verify file names contain corner position keywords

**Error: "No images found"**
- Check the path to your marker directory
- Ensure files have `.png`, `.jpg`, `.jpeg`, or `.bmp` extensions

**Missing annotations**
- The system will create full-image annotations automatically
- To use custom bounding boxes, place `.txt` files in the annotations directory

## Next Steps

1. Place your 4 marker PNG files in `data/gate_markers/`
2. Create or update `data/marker_config.json` with the correct mapping
3. Run the import command above
4. Train your model with: `python src/training/train.py --data data/aruco_dataset/dataset.yaml`

The model will now learn to distinguish between the 4 different gate corner positions!
