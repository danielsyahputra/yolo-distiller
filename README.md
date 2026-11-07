# Ultralytics YOLO with Cross-Architecture Knowledge Distillation (`YOLO26` / `YOLO12` / `YOLO11` / `YOLOv10` / `YOLOv9` / `YOLOv8`)

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/events/yolovision" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO Vision banner">
      </a>
  </p>
</div>

## Core Idea

![image](https://github.com/user-attachments/assets/2acfbc67-7cf0-4591-ba27-cdf4370b88e2)

Supports **Same-Architecture** and **Cross-Architecture** Knowledge Distillation across all modern Ultralytics YOLO families (**YOLO26**, **YOLO12**, **YOLO11**, **YOLOv10**, **YOLOv9**, and **YOLOv8**) and tasks (`Detect`, `Segment` / `Segment26`, `Pose` / `Pose26`, `OBB` / `OBB26`):

- **`cwd` (Channel-Wise Distillation — Default, Recommended for Segmentation & Detection):** Converts each of the 6 multi-scale Backbone + Neck feature maps (`[B_P4, B_P5, N_td_P4, N_P3, N_P4, N_P5]`) into spatial probability distributions via channel-wise softmax and minimizes KL divergence.
- **`mgd` (Masked Generative Distillation):** Randomly masks student spatial pixels and trains a $3\times 3$ generative block to reconstruct the teacher's full feature maps across all 6 Backbone + Neck stages.
- **`sl2` (Score-Weighted L2 Distillation):** Weights Neck feature MSE by the teacher detection head's spatial confidence scores so background noise is suppressed.

## How to Install & Run

```bash
git clone https://github.com/pranta-barua007/yolo-distiller.git
cd yolo-distiller
pip install -e .
```

### Example 1: Same-Architecture Distillation (`YOLO26l-seg` $\to$ `YOLO26n-seg`)

```python
from ultralytics import YOLO

teacher = YOLO("yolo26l-seg.pt")  # or path to fine-tuned teacher weights (best.pt)
student = YOLO("yolo26n-seg.pt")  # or path to student weights (best.pt)

student.train(
    data="<data-path>",
    teacher=teacher.model,        # accepts `teacher.model`, `teacher` (YOLO object), or path to `.pt`
    distillation_loss="cwd",      # "cwd" | "mgd" | "sl2"
    imgsz=640,
    epochs=100,
    batch=16,
    workers=2,
    exist_ok=True,
)
```

### Example 2: Cross-Architecture Distillation (`YOLO11l-seg` $\leftrightarrow$ `YOLO26n-seg`, `YOLO12` $\leftrightarrow$ `YOLO26`, etc.)

```python
from ultralytics import YOLO

# Distill from a fine-tuned YOLO11 teacher into a YOLO26 student (or vice versa!)
teacher = YOLO("yolo11l-seg-best.pt")
student = YOLO("yolo26n-seg.pt")

student.train(
    data="<data-path>",
    teacher=teacher.model,
    distillation_loss="cwd",      # "cwd" | "mgd" | "sl2"
    imgsz=640,
    epochs=100,
    batch=16,
)
```

## Credits

- Channel-Wise Distillation Loss Implementation: [https://github.com/pppppM/mmdetection-distiller](https://github.com/pppppM/mmdetection-distiller)
- Mask Generation Distillation Loss Implementation: [https://github.com/yzd-v/MGD](https://github.com/yzd-v/MGD)
- YOLOv5 Knowledge Distillation Implementation: [https://github.com/wonbeomjang/yolov5-knowledge-distillation](https://github.com/wonbeomjang/yolov5-knowledge-distillation)
