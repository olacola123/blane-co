# Oppgave 1: Object Detection — Mathea

## Hva jeg har prøvd
| # | Tilnærming | Score | Beholdt? | Notater |
|---|-----------|-------|----------|---------|
| 1 | YOLOv8m, 100 epochs, 248 bilder, ONNX export | 0.64 | Ja | Trent på Colab T4, eksportert til ONNX for torch-kompatibilitet |

## Nåværende strategi
YOLOv8m trent på alle 248 treningsbilder (COCO-format konvertert til YOLO). Eksportert til ONNX (opset 17) for å unngå torch-versjonsmismatch mellom Colab (2.10) og sandbox (2.6).

## Neste steg
- Bruk NM_NGD_product_images til å booste klassifisering (30%-delen av scoren)
- Prøv YOLOv8l (større modell) for bedre detection
- Øk imgsz til 1280 for å fange små produkter bedre
- Legg til product images som ekstra treningsdata

## Funn
- Sandbox kjører torch 2.6.0 — tren/eksporter med ONNX (opset 17) for å unngå versjonsmismatch
- `files.upload()` i Colab krasjer på filer > ~500MB — bruk Google Drive mount i stedet
- zip på Mac inkluderer __MACOSX-filer — bruk `-x "*.DS_Store" -x "__MACOSX/*"` flagg
