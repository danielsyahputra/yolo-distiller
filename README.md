# Ultralytics YOLO with Additional Knowledge Distillation Capability (YOLO26 / YOLO12 / YOLO11 / YOLOv10 / YOLOv9 / YOLOv8)

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/events/yolovision" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO Vision banner">
      </a>
  </p>
</div>

## Core Idea

![image](https://github.com/user-attachments/assets/2acfbc67-7cf0-4591-ba27-cdf4370b88e2)

Supports feature-based Knowledge Distillation (`cwd` - Channel-Wise Distillation, and `mgd` - Masked Generative Distillation) across **YOLO26** (`Detect`, `Segment26`, `Pose26`, `OBB26`), **YOLO12**, **YOLO11**, **YOLOv10**, **YOLOv9**, and **YOLOv8**.

## How to Install & Run

```bash
git clone https://github.com/pranta-barua007/yolo-distiller.git
cd yolo-distiller
pip install -e .
```

### Example (YOLO26 Instance Segmentation / Detection Knowledge Distillation)

```python
from ultralytics import YOLO

teacher = YOLO("yolo26l-seg.pt")  # or path to fine-tuned teacher weights (best.pt)
student = YOLO("yolo26n-seg.pt")  # or path to student weights (best.pt)

student.train(
    data="<data-path>",
    teacher=teacher.model,        # or pass `teacher` (YOLO object) or path to `.pt` directly
    distillation_loss="cwd",      # "cwd" (Channel-Wise Distillation) or "mgd" (Masked Generative Distillation)
    imgsz=1024,
    epochs=100,
    batch=16,
    workers=2,
    exist_ok=True,
)
```

## Credits

- Channel-Wise Distillation Loss Implementation: [https://github.com/pppppM/mmdetection-distiller](https://github.com/pppppM/mmdetection-distiller)
- Mask Generation Distillation Loss Implementation: [https://github.com/yzd-v/MGD](https://github.com/yzd-v/MGD)
- YOLOv5 Knowledge Distillation Implementation: [https://github.com/wonbeomjang/yolov5-knowledge-distillation](https://github.com/wonbeomjang/yolov5-knowledge-distillation)
