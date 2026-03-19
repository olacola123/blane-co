# Oppgave 1: NorgesGruppen Data — Object Detection

## Beskrivelse
Tren en object detection-modell for å identifisere dagligvareprodukter på norske butikkhyller. Submit kode som ZIP-fil som kjøres i sandboxed Docker-container med GPU.

## Scoring: Hybrid mAP@0.5
- **Detection mAP (70%)**: Bounding box-nøyaktighet, kategori ignorert (IoU >= 0.5)
- **Classification mAP (30%)**: Riktig produktidentifikasjon (IoU >= 0.5 OG riktig category_id)
- Detection-only submissions kan score opptil 70%
- Score-range: 0.0 til 1.0

## Dataset

### COCO Dataset (~864 MB)
- 248 hyllebilder fra norske dagligvarebutikker
- ~22,700 COCO-format bounding box-annotasjoner
- 356 produktkategorier (category_id 0-355)
- Bilder fra 4 butikkseksjoner: Egg, Frokost, Knekkebrod, Varmedrikker

### Produkt-referansebilder (~60 MB)
- 327 individuelle produkter med multi-angle-bilder
- Organisert etter strekkode med main, front, back, left, right, top, bottom views
- Inkluderer metadata.json med produktnavn og annotasjonstall

### Annotasjonsformat (COCO)
annotations.json inneholder images, categories, og annotations arrays. Hver annotasjon:
- `bbox`: [x, y, width, height] i COCO pixel-format
- `product_code`: strekkode-identifikator
- `category_id`: integer 0-355 som mapper til produktnavn
- `corrected`: boolean for manuell verifisering

## Submission-format

### ZIP-struktur
- `run.py` MÅ ligge i roten av ZIP-filen (IKKE i en undermappe)
- Maks 420 MB total størrelse
- Inkluder modellvekter og hjelpefiler i Python

### run.py-kontrakt
```bash
python run.py --input /data/images/ --output predictions.json
```

### Output JSON
Array med prediksjoner, hver med:
- `image_id` — bilde-ID
- `bbox` — [x, y, width, height]
- `category_id` — produkt-kategori (0-355)

## Sandbox-miljø
- **GPU**: NVIDIA L4 (24 GB VRAM)
- **Pre-installert**: PyTorch 2.6.0, YOLOv8, ONNX Runtime, OpenCV, scikit-learn
- **Ingen nettverkstilgang**, ingen pip install ved kjøretid
- **Timeout**: 300 sekunder per submission
- **Sikkerhet**: blokkerer os/subprocess imports, eval/exec, symlinks

## Constraints
- 356 produktkategorier (ID 0-355)
- FP16-kvantisering anbefalt for modellstørrelse
- ONNX-eksport støttet for alle rammeverk
- Aldri assign probability 0.0 til noen klasse

## Strategi-tips
- Start med detection-only (YOLOv8) for å nå 70%-taket raskt
- Legg til klassifisering etterpå for de siste 30%
- Bruk referansebildene for augmentation/few-shot
- FP16 + ONNX for å holde modellen under 420 MB
