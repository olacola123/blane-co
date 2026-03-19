# Oppgave 1: NorgesGruppen Data — Object Detection

## Hvordan det fungerer
1. Last ned treningsdata fra konkurranse-nettsiden (krever innlogging)
2. Tren object detection-modell lokalt
3. Skriv `run.py` som tar hyllebilder og gir prediksjoner
4. Zip kode + modellvekter
5. Last opp på submit-siden
6. Kjøres i sandbox med GPU (NVIDIA L4, 24GB VRAM), ingen nettverkstilgang
7. Scoring: 70% detection + 30% classification

## Treningsdata

### COCO Dataset (~864 MB)
- 248 hyllebilder fra norske dagligvarebutikker
- ~22,700 COCO-format bounding box-annotasjoner
- 356 produktkategorier (category_id 0-355)
- 4 butikkseksjoner: Egg, Frokost, Knekkebrod, Varmedrikker

### Produkt-referansebilder (~60 MB)
- 327 individuelle produkter med multi-angle-bilder
- Organisert etter strekkode: `{product_code}/main.jpg`, `front.jpg`, etc.
- Inkluderer `metadata.json` med produktnavn og annotasjonstall

### Annotasjonsformat (COCO)
```
bbox: [x, y, width, height]   # pixels
category_id: 0-355
product_code: barcode string
corrected: boolean             # manuelt verifisert
```

## run.py-kontrakt

```bash
python run.py --input /data/images --output /output/predictions.json
```

### Output JSON
```json
[
  {"image_id": 42, "category_id": 0, "bbox": [x, y, w, h], "score": 0.923}
]
```
- `image_id` — fra filnavn (`img_00042.jpg` → `42`)
- `category_id` — 0-355
- `bbox` — COCO-format `[x, y, width, height]`
- `score` — 0-1

## Scoring: Hybrid mAP@0.5

```
Score = 0.7 x detection_mAP + 0.3 x classification_mAP
```

- **Detection mAP**: IoU >= 0.5, kategori ignorert
- **Classification mAP**: IoU >= 0.5 OG riktig `category_id`
- Detection-only (alle `category_id: 0`) gir maks 0.70

## Submission-begrensninger
- 2 in-flight per lag
- 3 per dag per lag
- 2 infrastructure failure freebies per dag
- Reset ved midnatt UTC

## ZIP-struktur

```
submission.zip
├── run.py          ← MÅ ligge i ROTEN (ikke i undermappe!)
├── model.pt        ← modellvekter
├── config.yaml     ← valgfritt
└── utils.py        ← valgfritt
```

### Begrensninger
- Maks 420 MB ukomprimert
- Maks 1000 filer, 10 `.py`-filer, 3 weight-filer
- Tillatte filtyper: `.py`, `.json`, `.yaml`, `.yml`, `.cfg`, `.pt`, `.pth`, `.onnx`, `.safetensors`, `.npy`

### Lag ZIP riktig
```bash
cd my_submission/
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
# Verifiser:
unzip -l submission.zip | head -10
```

## Sandbox-miljø

| Ressurs | Spesifikasjon |
|---------|---------------|
| GPU | NVIDIA L4, 24 GB VRAM |
| CPU | 4 vCPU |
| RAM | 8 GB |
| Python | 3.11 |
| CUDA | 12.4 |
| Nettverk | Ingen |
| Timeout | 300 sekunder |

`torch.cuda.is_available()` → `True`

### Pre-installerte pakker
| Pakke | Versjon |
|-------|---------|
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| ultralytics | 8.1.0 |
| onnxruntime-gpu | 1.20.0 |
| opencv-python-headless | 4.9.0.80 |
| albumentations | 1.3.1 |
| Pillow | 10.2.0 |
| numpy | 1.26.4 |
| scipy | 1.12.0 |
| scikit-learn | 1.4.0 |
| pycocotools | 2.0.7 |
| ensemble-boxes | 1.0.9 |
| timm | 0.9.12 |
| safetensors | 0.4.2 |
| supervision | 0.18.0 |

### Modeller tilgjengelige i sandbox
- **ultralytics 8.1.0**: YOLOv8n/s/m/l/x, YOLOv5u, RT-DETR-l/x
- **torchvision 0.21.0**: Faster R-CNN, RetinaNet, SSD, FCOS, Mask R-CNN
- **timm 0.9.12**: ResNet, EfficientNet, ViT, Swin, ConvNeXt (backbones)

### IKKE i sandbox (bruk ONNX eller inkluder kode)
YOLOv9, YOLOv10, YOLO11, RF-DETR, Detectron2, MMDetection, HuggingFace Transformers

## Versjonskompatibilitet — VIKTIG

| Problem | Konsekvens |
|---------|------------|
| ultralytics 8.2+ weights på 8.1.0 | Feiler. Pin 8.1.0 ved trening |
| torch 2.7+ full save på 2.6.0 | Kan feile. Bruk `state_dict` |
| timm 1.0+ på 0.9.12 | Feiler. Pin 0.9.12 |
| ONNX opset > 20 | Feiler. Bruk opset 17 |

## Sikkerhetsrestriksjoner

**Blokkert:** `os`, `subprocess`, `socket`, `ctypes`, `builtins`, `eval()`, `exec()`, `compile()`, `__import__()`

Bruk `pathlib` i stedet for `os`.

## Vanlige feil

| Feil | Årsak | Fix |
|------|-------|-----|
| `run.py not found` | ZIP inneholder mappe, ikke filer | Zip innholdet, ikke mappen |
| `__MACOSX` | Finder-zip | Bruk terminal `zip`-kommando |
| `.bin disallowed` | Ugyldig filtype | Rename til `.pt` eller bruk safetensors |
| Security violation | `os`/`subprocess` imports | Bruk `pathlib` |
| No `predictions.json` | Skriver ikke til `--output` path | Bruk `argparse` output path |
| Timeout 300s | For treg modell | Bruk GPU, mindre modell |
| Exit 137 | OOM | Reduser batch size eller bruk FP16 |
| Exit 139 | Segfault | Versjonsmismatch |
| `ModuleNotFoundError` | Pakke ikke i sandbox | Bruk ONNX eller inkluder kode |
| `KeyError on load` | Versjonsmismatch | Sjekk kompatibilitetstabell |

## Tips
- Start med random baseline for å verifisere setup
- GPU tilgjengelig — larger models er mulig
- FP16 anbefalt (mindre + raskere)
- ONNX+CUDA bra for alle rammeverk
- Prosesser bilder en om gangen for minnehåndtering
- `torch.no_grad()` under inference

## Eksempelkode: Random Baseline

Verifiserer at submission-pipeline fungerer.

```python
"""run.py — Random baseline for testing submission pipeline."""
import argparse
import json
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    predictions = []

    for img_path in sorted(input_dir.glob("*.jpg")):
        # Hent image_id fra filnavn: img_00042.jpg → 42
        image_id = int(img_path.stem.split("_")[-1])

        # Generer 5 tilfeldige bboxer per bilde
        for _ in range(5):
            predictions.append({
                "image_id": image_id,
                "category_id": random.randint(0, 355),
                "bbox": [
                    random.uniform(0, 800),   # x
                    random.uniform(0, 600),   # y
                    random.uniform(50, 200),  # width
                    random.uniform(50, 200),  # height
                ],
                "score": random.uniform(0.1, 1.0),
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {output_path}")

if __name__ == "__main__":
    main()
```

## Eksempelkode: YOLOv8 Baseline

Tren lokalt med `ultralytics==8.1.0`, submit vektene.

```python
"""run.py — YOLOv8 inference for shelf product detection."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)

    # Last modell (best.pt må ligge i ZIP-roten)
    model = YOLO("best.pt")

    predictions = []

    for img_path in sorted(input_dir.glob("*.jpg")):
        image_id = int(img_path.stem.split("_")[-1])

        # Inference med GPU
        results = model.predict(
            str(img_path),
            device="cuda",
            conf=0.25,
            iou=0.45,
            half=True,       # FP16
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(boxes.cls[i].item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # xyxy → xywh
                    "score": float(boxes.conf[i].item()),
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {output_path}")

if __name__ == "__main__":
    main()
```

### Treningseksempel (lokalt, IKKE i submission)
```python
"""train.py — Tren YOLOv8 lokalt. Pin ultralytics==8.1.0!"""
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # Pretrained medium
model.train(
    data="dataset.yaml",     # Pek til COCO-datasettet
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    half=True,
    name="shelf_detect",
)
# best.pt havner i runs/detect/shelf_detect/weights/best.pt
```
