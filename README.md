# CAT-Seg 病虫害语义分割实验复现记录
## 1. 项目简介

本项目基于 CAT-Seg 模型完成农业病虫害图像语义分割任务。

目标：

输入：农作物病虫害图片
输出：像素级病虫害区域分割结果

数据集包含：

26 类病虫害
1 类背景

共: 27classes

本实验数据集共包含27个语义类别，其中0表示背景类别(background)，1-26表示不同病虫害类别。

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

## 2. 实验环境
### 服务器配置
  本实验模型运行于Autodl平台vGPU-32GB服务器
  | 项目 | 配置|
  |-----|-----|
  | CPU | 12 vCPU Intel Xeon Platinum 8352V |
  | GPU | NVIDIA RTX4080 32GB |
  | Ram | 62GB |
  | CUDA | 13.0 |
  | Python | 3.8.20 |

## 3. 基础环境安装
### 3.1 环境配置
  ```bash
# 创建环境
conda create -n catseg python=3.8

 # 初始化 conda 到 bash (Autodl)
 conda init bash

 # 重载配置 (Autodl)
 source /root/.bashrc

# 激活环境
conda activate catseg

# 进入项目目录
cd ~/autodl-tmp/CAT-Seg
```
### 3.2 PyTorch环境
运行
```bash
python -c "import torch;print(torch.__version__)"
```
输出
```text
2.4.1+cu118
```


### 3.3 获取CAT-Seg代码

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
### 3.4 配置环境依赖
运行
```bash
pip install -r requirements.txt
```

requirements:
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
### 3.5 open_clip配置
  CAT-Seg 需要 open_clip，因为 CAT-Seg 本身依赖 CLIP（Contrastive Language-Image Pre-training）进行开放词汇语义分割（Open-Vocabulary Semantic Segmentation）。
  ```text
                图片
                 ↓
          Image Encoder
                 |
             图像特征
                 |
                 ↓
            相似度计算
                 ↑
           |Text Encoder|
                 |
        "apple black rot"
        "corn rust"
        "aphids"
```
安装
```bash
pip install open_clip_torch
```
验证
```bash
python -c "import open_clip"
```

## 4. 数据集处理
### 4.1 原始数据格式
Labelme格式:
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
json示例:
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
因此我们要转换为CAT-Seg所需求格式

### 4.2 格式转换
```text
路径: ./dataset5
```
#### 4.2.1 生成类别文件
确定类的数量，统计类别
创建：
```bash
count_classes.py
```
count_classes.py
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
运行:
```bash
python get_classes.py
```
Output

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
#### 4.2.2 Labelme → CAT-Seg mask转换
安装：
```bash
pip install pillow numpy opencv-python
```
创建:
```text
exchange.py
```
exchange.py
```python
import os
import json
import cv2
import numpy as np
from PIL import Image

# 原始数据
src_root=""

# 输出目录
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

# 类别编号
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
        # 保存mask

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
        # 复制图片

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
运行:
```bash
python exchange.py
```
输出:
```text
Done~
```

现在整体目录结构
```text
autodl-tmp
├── CAT_Seg
|
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
          │            ├
          │            └── classes.txt
          │
          ├── count_classes.py
          │
          ├── exchange.py
```






