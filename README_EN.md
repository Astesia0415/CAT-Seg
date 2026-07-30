# CAT-Seg Based Semantic Segmentation of Crop Pests and Diseases

**Read this in other languages: [English](README_EN.md), [中文](README.md).**

## 1. Project Introduction

This project performs semantic segmentation of agricultural pest and disease images based on the CAT-Seg model.

**Goal:**

Input: Crop pest and disease images
Output: Pixel-level segmentation of pest and disease regions

**Dataset includes:**

- 26 pest and disease classes
- 1 background class

Total: 27 classes

The experimental dataset contains 27 semantic categories, where 0 represents the background class, and 1–26 represent different pest and disease categories.

| ID | Category | ID | Category |
|----|----------|----|----------|
| 0 | background | 13 | bell pepper blossom end rot |
| 1 | Aleurocanthus spiniferus | 14 | citrus canker |
| 2 | Ceroplastes rubens | 15 | corn gray leaf spot |
| 3 | Icerya purchasi Maskell | 16 | corn rust |
| 4 | Limacodidae | 17 | corn smut |
| 5 | Locustoidea | 18 | garlic rust |
| 6 | Potosiabre vitarsis | 19 | legume blister beetle |
| 7 | alfalfa plant bug | 20 | oides decempunctata |
| 8 | aphids | 21 | rice blast |
| 9 | apple black rot | 22 | tarnished plant bug |
| 10 | banana anthracnose | 23 | wheat leaf rust |
| 11 | banana black leaf streak | 24 | wheat loose smut |
| 12 | bean halo blight | 25 | wheat septoria blotch |
| 26 | wheat stripe rust | | |

## 2. Experimental Environment

### Server Configuration

This experiment runs on the AutoDL platform vGPU-32GB server.

| Item | Specification |
|------|---------------|
| OS | Ubuntu 20.04.3 LTS |
| CPU | 12 vCPU Intel Xeon Platinum 8352V |
| GPU | NVIDIA RTX 4080 32GB |
| RAM | 62GB |
| CUDA | 13.0 |
| Python | 3.8.20 |

### Auxiliary Device Configuration

| Item | Specification |
|------|---------------|
| OS | Windows 10 Professional 22H2 |
| CPU | Intel Core Ultra 9 275HX 2.70 GHz |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU 8GB |
| RAM | 16GB |
| CUDA | 13.1 |
| Python | 3.10.20 |

## 3. Basic Environment Setup

### 3.1 Environment Configuration

```bash
# Create environment
conda create -n catseg python=3.8

# Initialize conda for bash (AutoDL)
conda init bash

# Reload configuration (AutoDL)
source /root/.bashrc

# Activate environment
conda activate catseg

# Enter project directory
cd ~/autodl-tmp/CAT-Seg
```

### 3.2 PyTorch Environment

**Install:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

**Run:**
```bash
python -c "import torch;print(torch.__version__)"
```

**Output:**
```text
2.4.1+cu118
```

### 3.3 Obtain CAT-Seg Code

```bash
git clone https://github.com/cvlab-kaist/CAT-Seg.git

cd CAT-Seg
```

```text
CAT-Seg
│
├── cat_seg
├── configs
├── datasets
├── train_net.py
├── eval.sh
└── requirements.txt
```

### 3.4 Install Environment Dependencies

**Run:**
```bash
pip install -r requirements.txt
```

**requirements:**
```text
scipy
ftfy
opencv-python
setuptools
pillow
imageio
timm
regex
einops
```

### 3.5 open_clip Configuration

CAT-Seg requires open_clip because CAT-Seg itself depends on CLIP (Contrastive Language-Image Pre-training) for open-vocabulary semantic segmentation.

```text
                Image
                  ↓
           Image Encoder
                  |
           Image Features
                  |
                  ↓
          Similarity Computation
                  ↑
            |Text Encoder|
                  |
         "apple black rot"
         "corn rust"
         "aphids"
```

**Install:**
```bash
pip install open_clip_torch
```

**Verify:**
```bash
python -c "import open_clip"
```

## 4. Dataset Processing

**Working directory:**
```text
./autodl-tmp/dataset5
```

### 4.1 Original Data Format

Labelme format:
```text
dataset5
│
├── train
│   ├── images
│   └── labels
│
├── val
│   ├── images
│   └── labels
│
└── test
    ├── images
    └── labels
```

**JSON example:**
```text
{
"label":"apple black rot",
"shape_type":"polygon",
"points":[
 [203.26,164.89],
 [188.72,185.59]
]
}
```

Therefore, we need to convert it to the format required by CAT-Seg.

### 4.2 Format Conversion

**Working directory:**
```text
./autodl-tmp/dataset5
```

#### 4.2.1 Generate Class File

Determine the number of classes and collect class statistics.

**Create:**
```bash
count_classes.py
```

<details>
  <summary>count_classes.py</summary>

```python
import os
import json

root = ""

classes = set()

for split in ["train","val","test"]:

    label_dir = os.path.join(
        root,
        split,
        "labels"
    )

    for file in os.listdir(label_dir):

        if file.endswith(".json"):

            path=os.path.join(
                label_dir,
                file
            )

            with open(path,"r",encoding="utf-8") as f:
                data=json.load(f)

            for obj in data["shapes"]:
                classes.add(obj["label"])

print("classes:")

for i,c in enumerate(sorted(classes)):

    print(i+1,c)

```

</details>

**Run:**
```bash
python get_classes.py
```

<details>

<summary>Output</summary>

```text
classes:
1 Aleurocanthus spiniferus
2 Ceroplastes rubens
3 Icerya purchasi Maskell
4 Limacodidae
5 Locustoidea
6 Potosiabre vitarsis
7 alfalfa plant bug
8 aphids
9 apple black rot
10 banana anthracnose
11 banana black leaf streak
12 bean halo blight
13 bell pepper blossom end rot
14 citrus canker
15 corn gray leaf spot
16 corn rust
17 corn smut
18 garlic rust
19 legume blister beetle
20 oides decempunctata
21 rice blast
22 tarnished plant bug
23 wheat leaf rust
24 wheat loose smut
25 wheat septoria blotch
26 wheat stripe rust
```

</details>

#### 4.2.2 Labelme → CAT-Seg Mask Conversion

**Install:**
```bash
pip install pillow numpy opencv-python
```

**Create:**
```text
exchange.py
```

<details>
  <summary>exchange.py</summary>

```python
import os
import json
import cv2
import numpy as np
from PIL import Image

# Original data
src_root=""

# Output directory
out_root="CATSeg_dataset"

classes=[
    "Aleurocanthus spiniferus",
    "Ceroplastes rubens",
    "Icerya purchasi Maskell",
    "Limacodidae",
    "Locustoidea",
    "Potosiabre vitarsis",
    "alfalfa plant bug",
    "aphids",
    "apple black rot",
    "banana anthracnose",
    "banana black leaf streak",
    "bean halo blight",
    "bell pepper blossom end rot",
    "citrus canker",
    "corn gray leaf spot",
    "corn rust",
    "corn smut",
    "garlic rust",
    "legume blister beetle",
    "oides decempunctata",
    "rice blast",
    "tarnished plant bug",
    "wheat leaf rust",
    "wheat loose smut",
    "wheat septoria blotch",
    "wheat stripe rust"
]

# Class mapping
class_map={
    name:i+1
    for i,name in enumerate(classes)
}

for split in ["train","val","test"]:

    img_dir=os.path.join(
        src_root,
        split,
        "images"
    )

    label_dir=os.path.join(
        src_root,
        split,
        "labels"
    )

    out_img_dir=os.path.join(
        out_root,
        "images",
        split
    )

    out_mask_dir=os.path.join(
        out_root,
        "masks",
        split
    )

    os.makedirs(
        out_img_dir,
        exist_ok=True
    )

    os.makedirs(
        out_mask_dir,
        exist_ok=True
    )

    for json_file in os.listdir(label_dir):

        if not json_file.endswith(".json"):
            continue

        json_path=os.path.join(
            label_dir,
            json_file
        )

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data=json.load(f)

        h=data["imageHeight"]
        w=data["imageWidth"]

        mask=np.zeros(
            (h,w),
            dtype=np.uint8
        )

        for shape in data["shapes"]:

            label=shape["label"]

            if label not in class_map:
                continue

            points=np.array(
                shape["points"],
                dtype=np.int32
            )

            cv2.fillPoly(
                mask,
                [points],
                class_map[label]
            )
        # Save mask

        name=os.path.splitext(
            json_file
        )[0]

        mask_path=os.path.join(
            out_mask_dir,
            name+".png"
        )

        Image.fromarray(mask).save(
            mask_path
        )
        # Copy image

        img_name=data["imagePath"]

        src_img=os.path.join(
            img_dir,
            img_name
        )

        dst_img=os.path.join(
            out_img_dir,
            img_name
        )

        if os.path.exists(src_img):

            Image.open(src_img).save(
                dst_img
            )
print("Done!")
```

</details>

**Run:**
```bash
python exchange.py
```

**Output:**
```text
Done~
```

**Create classes.txt**

<details>
  <summary>classes.txt</summary>

```text
background
Aleurocanthus spiniferus
Ceroplastes rubens
Icerya purchasi Maskell
Limacodidae
Locustoidea
Potosiabre vitarsis
alfalfa plant bug
aphids
apple black rot
banana anthracnose
banana black leaf streak
bean halo blight
bell pepper blossom end rot
citrus canker
corn gray leaf spot
corn rust
corn smut
garlic rust
legume blister beetle
oides decempunctata
rice blast
tarnished plant bug
wheat leaf rust
wheat loose smut
wheat septoria blotch
wheat stripe rust
```

</details>

**Create mask visualization script:**
check_mask.py

*(Note: modify the image path as needed)*

<details>
  <summary>check_mask.py</summary>

```python
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# Modify this to use an existing image path
img_path = r"CATSeg_dataset/images/train/apple_black_rot_1_jpg.rf.e21dee4a83b8d37c916ed9d2dc1b2896.jpg"

mask_path = r"CATSeg_dataset/masks/train/apple_black_rot_1_jpg.rf.e21dee4a83b8d37c916ed9d2dc1b2896.png"


img = Image.open(img_path)

mask = np.array(
    Image.open(mask_path)
)


plt.figure(figsize=(12,5))


plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Image")
plt.axis("off")


plt.subplot(1,2,2)
plt.imshow(mask)
plt.title("Mask")
plt.axis("off")


plt.show()


print("Mask classes:", np.unique(mask))
```

</details>

**Output**

![output](asset/picture/checkmask.png)

```text

Mask classes: [0 x]

```

Where `x` is the class number 1–26 from classes.txt.
For example:
```text
Mask classes: [0 9]
```
Indicates: 9 = apple black rot

## Server Directory Structure

**Working directory:**
```text
./autodl-tmp/CAT-seg
```

**Overall directory structure:**
```text
autodl-tmp
├── CAT_Seg
|         │
│         ├── cat_seg
│         ├── configs
│         ├── datasets
│         ├── train_net.py
│         ├── eval.sh
│         └── requirements.txt
│
└── dataset5
          ├── CATSeg_dataset
          │            │
          │            ├── images
          │            │   ├── train
          │            │   ├── val
          │            │   └── test
          │            │
          │            ├── masks
          │            │   ├── train
          │            │   ├── val
          │            │   └── test
          │            │
          │            └── classes.txt
          │
          ├── count_classes.py
          │
          ├── exchange.py
          │
          └── check_mask.py
```

## 5. Dataset Registration

Detectron2 only supports public datasets like COCO, VOC, ADE20K by default.
Therefore, in:

```text
CAT-Seg
│
└── cat_seg
    │
    └── data
        │
        └── datasets
            │
            └── register_pest.py
```

**Create:**
```text
register_pest.py
```

<details>
  <summary>register_pest.py</summary>

```python
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg


_PREDEFINED_SPLITS_PEST = {
    "pest_train": (
        "/root/autodl-tmp/dataset5/CATSeg_dataset/images/train",
        "/root/autodl-tmp/dataset5/CATSeg_dataset/masks/train",
    ),

    "pest_val": (
        "/root/autodl-tmp/dataset5/CATSeg_dataset/images/val",
        "/root/autodl-tmp/dataset5/CATSeg_dataset/masks/val",
    ),

    "pest_test": (
        "/root/autodl-tmp/dataset5/CATSeg_dataset/images/test",
        "/root/autodl-tmp/dataset5/CATSeg_dataset/masks/test",
    ),
}


def register_all_pest(root=None):

    for name, (image_dir, gt_dir) in _PREDEFINED_SPLITS_PEST.items():

        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir:
            load_sem_seg(
                y,
                x,
                gt_ext="png",
                image_ext="jpg"
            )
        )


        MetadataCatalog.get(name).set(
            ignore_label=255,
            evaluator_type="sem_seg",
            stuff_classes=[
                "background",
                "Aleurocanthus spiniferus",
                "Ceroplastes rubens",
                "Icerya purchasi Maskell",
                "Limacodidae",
                "Locustoidea",
                "Potosiabre vitarsis",
                "alfalfa plant bug",
                "aphids",
                "apple black rot",
                "banana anthracnose",
                "banana black leaf streak",
                "bean halo blight",
                "bell pepper blossom end rot",
                "citrus canker",
                "corn gray leaf spot",
                "corn rust",
                "corn smut",
                "garlic rust",
                "legume blister beetle",
                "oides decempunctata",
                "rice blast",
                "tarnished plant bug",
                "wheat leaf rust",
                "wheat loose smut",
                "wheat septoria blotch",
                "wheat stripe rust",
            ]
        )

register_all_pest()

```

</details>

### Import the Registration File

Modify:
```text
cat_seg/data/datasets/__init__.py
```

Add:
```python
from .register_pest import register_all_pest
```

Purpose: Allows CAT-Seg to find the custom dataset when loading the data module.

### Call Registration Function at Training Entry Point

Modify:
```text
./autodl-tmp/CAT-Seg/train_net.py
```

Add inside `main()`:
```python
register_all_pest()
```

### Verify Registration Success

**Create:**
```text
test_dataset.py
```

<details>
  <summary>test_dataset.py</summary>

```python
from detectron2.data import DatasetCatalog
import cat_seg.data.datasets


print(
    DatasetCatalog.get("pest_train")[:1]
)
```

</details>

On the first run, if registration is successful, you will see output like:

```text
[
{
'file_name':
'/root/autodl-tmp/dataset5/CATSeg_dataset/images/train/IP024000016.jpg',

'sem_seg_file_name':
'/root/autodl-tmp/dataset5/CATSeg_dataset/masks/train/IP024000016.png'
}
]
```

This indicates that images and masks are correctly paired.

On subsequent runs, an error will appear:

```text
AssertionError: Dataset 'pest_train' is already registered!

```

## 6. Training Configuration

### 6.1 Create Class File

**Create:**
```text
datasets/pest27.json
```

<details>
  <summary>pest27.json</summary>

```json
[
"background",
"Aleurocanthus spiniferus",
"Ceroplastes rubens",
"Icerya purchasi Maskell",
"Limacodidae",
"Locustoidea",
"Potosiabre vitarsis",
"alfalfa plant bug",
"aphids",
"apple black rot",
"banana anthracnose",
"banana black leaf streak",
"bean halo blight",
"bell pepper blossom end rot",
"citrus canker",
"corn gray leaf spot",
"corn rust",
"corn smut",
"garlic rust",
"legume blister beetle",
"oides decempunctata",
"rice blast",
"tarnished plant bug",
"wheat leaf rust",
"wheat loose smut",
"wheat septoria blotch",
"wheat stripe rust"
]

```

</details>

### 6.2 Create Training Configuration

**Copy the original config:**
```bash
cd ~/autodl-tmp/CAT-Seg/configs

cp vitb_384.yaml pest_vitb_384.yaml
```

**Modify in `pest_vitb_384.yaml`**, changing:

```yaml
SEM_SEG_HEAD:
  NUM_CLASSES: 171
  TRAIN_CLASS_JSON: "datasets/coco.json"
  TEST_CLASS_JSON: "datasets/coco.json"
```

to:

```yaml
SEM_SEG_HEAD:
  NUM_CLASSES: 27
  TRAIN_CLASS_JSON: "datasets/pest27.json"
  TEST_CLASS_JSON: "datasets/pest27.json"
```

**Add dataset name** at the end of `pest_vitb_384.yaml`:

```yaml
DATASETS:
  TRAIN: ("pest_train",)
  TEST: ("pest_val",)
```

**Adjust as needed:**
```yaml
MAX_ITER: 50000
```

## 7. Training

### Test Run

Run 100 iterations first to check: CLIP weights, dataloader, loss, GPU memory, mask size.

```bash
python train_net.py \
--config-file configs/pest_vitb_384.yaml \
SOLVER.MAX_ITER 100 \
--num-gpus 1
```

### Full Training

```bash
python train_net.py \
--config-file configs/pest_vitb_384.yaml \
--num-gpus 1
```

### Training Results

**Model evaluation metrics:**

CAT-Seg belongs to: semantic segmentation

```text
mIoU → Main semantic segmentation metric
fwIoU → Frequency-weighted IoU
pACC → Pixel accuracy
```

| Iteration | mIoU | fwIoU | mACC | pACC | Mask AP<sub>50</sub> | Mask AP<sub>50-95</sub> |
|----|----|----|----|----|----|----|
| 100 | 2.4589 | 3.5563 | 18.3378 | 5.8825 | unknown | unknown |
| 22000 | 59.6129 | 89.5649 | 72.5572 | 93.8348 | unknown | unknown |
| 50000 | unknown | unknown | unknown | unknown | unknown | unknown |

### Note: Mask AP Evaluation

Since this dataset provides semantic segmentation annotations, the semantic segmentation predictions are converted to instance masks using connected component analysis before COCO evaluation.

Therefore, the mask AP metrics reported in this document represent instance-level evaluation results derived from semantic segmentation prediction outputs.

## 8. Final Experiment Records

**Training output saved to:**
```text
output/
│
├── model_final.pth
├── log.txt
└── inference/
```

## Acknowledgements

This project is based on [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg).

Thanks to the CAT-Seg authors for their excellent work.

