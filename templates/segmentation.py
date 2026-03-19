"""
NM i AI 2026 — Segmentering Template
=====================================
Bilde-segmentering med U-Net og pretrained encoder.

Slik fungerer segmentering:
- Du har bilder + "masker" (fasit der hver piksel er merket)
- Modellen laerer a tegne riktig maske for nye bilder
- Evalueres med Dice score (overlap mellom prediksjon og fasit)

Bruk:
1. Legg bilder i images/ og masker i masks/
2. Tilpass data-lasting
3. Kjor: python segmentation.py

TODO-markerte steder ma tilpasses for den spesifikke oppgaven.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import json

# Sjekk GPU/MPS
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Bruker device: {DEVICE}")


# === LOSS FUNCTIONS ===

class DiceLoss(nn.Module):
    """
    Dice loss — maler overlap mellom prediksjon og fasit.
    Score 1.0 = perfekt match, 0.0 = ingen overlap.
    Vi minimerer (1 - Dice).
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Kombinasjon av Dice + Binary Cross Entropy.
    BCE hjelper med a laere piksler individuelt.
    Dice hjelper med a laere overlap globalt.
    Beste av begge verdener.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target)


class DiceFocalLoss(nn.Module):
    """
    Dice + Focal Loss — bedre for ubalanserte data (f.eks. liten tumor, stort bilde).
    Focal loss fokuserer pa vanskelige piksler modellen bommer pa.
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        # Focal loss
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p = torch.sigmoid(pred)
        focal_weight = self.alpha * target * (1 - p) ** self.gamma + (1 - self.alpha) * (1 - target) * p ** self.gamma
        focal = (focal_weight * bce).mean()

        return self.dice_weight * self.dice(pred, target) + self.focal_weight * focal


# === DATASET ===

class SegmentationDataset(Dataset):
    """
    Laster bilder og tilhorende masker.

    TODO: Tilpass til oppgavens dataformat.
    Forventet: images/001.png + masks/001.png (samme filnavn)
    """

    def __init__(self, image_dir: str, mask_dir: str, img_size: int = 256, augment: bool = False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augment = augment

        # Finn alle bilder
        self.image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npy"}
        ])
        print(f"Fant {len(self.image_paths)} bilder i {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # TODO: Tilpass mask-filnavn hvis det er annerledes
        mask_path = self.mask_dir / img_path.name

        # Last bilde
        if img_path.suffix == ".npy":
            image = np.load(img_path).astype(np.float32)
        else:
            image = np.array(Image.open(img_path).convert("RGB").resize(
                (self.img_size, self.img_size)
            )).astype(np.float32) / 255.0

        # Last maske
        if mask_path.suffix == ".npy":
            mask = np.load(mask_path).astype(np.float32)
        else:
            mask = np.array(Image.open(mask_path).convert("L").resize(
                (self.img_size, self.img_size)
            )).astype(np.float32) / 255.0

        # Terskel maske til binar (0 eller 1)
        mask = (mask > 0.5).astype(np.float32)

        # Data augmentation (gjor treningen mer robust)
        if self.augment:
            image, mask = self._augment(image, mask)

        # Konverter til tensorer
        if image.ndim == 2:
            image = image[np.newaxis, ...]  # (1, H, W) for grayscale
        elif image.ndim == 3:
            image = image.transpose(2, 0, 1)  # (C, H, W) for RGB

        mask = mask[np.newaxis, ...]  # (1, H, W)

        return torch.tensor(image), torch.tensor(mask)

    def _augment(self, image, mask):
        """Enkel data augmentation — gjor at modellen generaliserer bedre."""
        # Horisontal flip (50% sjanse)
        if np.random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Vertikal flip (50% sjanse)
        if np.random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()

        # Rotasjon 90/180/270 grader
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        # Lysstyrke-justeringer (kun bilde, ikke maske)
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 1)

        return image, mask


# === MODELL ===

def build_unet(in_channels=3, out_channels=1, encoder="resnet34", pretrained=True):
    """
    Bygg U-Net med pretrained encoder.

    Bruker segmentation_models_pytorch (smp) for enkel oppsett.
    Encoder er PRETRAINED pa ImageNet — allerede god til a "se" ting.
    Vi laerer den bare a segmentere var spesifikke oppgave.

    Populaere encoders:
    - resnet34: god balanse mellom speed og kvalitet (anbefalt start)
    - resnet50: litt bedre, litt tregere
    - efficientnet-b3: moderne, effektiv
    - mit_b2: transformer-basert, best for komplekse former
    """
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=out_channels,
        )
        print(f"U-Net med {encoder} encoder ({'pretrained' if pretrained else 'random'})")
        return model
    except ImportError:
        print("segmentation_models_pytorch ikke installert!")
        print("Kjor: pip install segmentation-models-pytorch")
        print("Bruker enkel U-Net i stedet...")
        return SimpleUNet(in_channels, out_channels)


class SimpleUNet(nn.Module):
    """
    Enkel U-Net uten avhengigheter — fungerer alltid.
    Bruk build_unet() med smp for bedre resultater.
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # Encoder (ned)
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(512, 1024)

        # Decoder (opp)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder med skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# === METRICS ===

def dice_score(pred, target, threshold=0.5):
    """Beregn Dice score — dette er sannsynligvis competition-metrikken."""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1) / (pred.sum() + target.sum() + 1)


def iou_score(pred, target, threshold=0.5):
    """Intersection over Union — alternativ metrikk."""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1) / (union + 1)


# === TRENING ===

def train(
    image_dir: str,
    mask_dir: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    img_size: int = 256,
    encoder: str = "resnet34",
    save_path: str = "models/segmentation",
    in_channels: int = 3,
):
    """
    Tren segmenteringsmodell.

    Args:
        image_dir: Mappe med treningsbilder
        mask_dir: Mappe med masker (fasit)
        epochs: Antall treningsrunder (50 er god start, 100+ for bedre)
        batch_size: Bilder per batch (senk hvis du gar tom for minne)
        lr: Laeringsrate (1e-4 er god standard)
        img_size: Bildestorrelse (256 er rask, 512 er bedre kvalitet)
        encoder: Pretrained encoder (resnet34, resnet50, efficientnet-b3)
        save_path: Hvor modellen lagres
    """
    print(f"=== Segmentering — Trening ===")
    print(f"  Bilder: {image_dir}")
    print(f"  Masker: {mask_dir}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}")
    print(f"  Bildestorrelse: {img_size}x{img_size}")

    # Data
    dataset = SegmentationDataset(image_dir, mask_dir, img_size=img_size, augment=True)

    # Split 80/20 train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Skru av augmentation for validering
    val_dataset.dataset.augment = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train: {train_size}, Val: {val_size}")

    # Modell
    model = build_unet(in_channels=in_channels, encoder=encoder).to(DEVICE)

    # Loss og optimizer
    criterion = DiceBCELoss()  # TODO: Bytt til DiceFocalLoss() for ubalanserte data
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Læringsrate-scheduler (senker LR gradvis for finere tuning)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Trening
    best_dice = 0
    Path(save_path).mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                val_dice += dice_score(outputs, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        scheduler.step()

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} — train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_dice: {val_dice:.4f}")

        # Lagre beste modell
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), f"{save_path}/best_model.pth")
            print(f"  -> Ny beste modell! Dice: {val_dice:.4f}")

    # Lagre siste modell
    torch.save(model.state_dict(), f"{save_path}/final_model.pth")
    print(f"\nTrening ferdig! Beste Dice: {best_dice:.4f}")
    print(f"Modeller lagret i {save_path}/")

    return model


# === PREDIKSJON ===

def predict(
    model_path: str,
    image_paths: list[str],
    img_size: int = 256,
    threshold: float = 0.5,
    encoder: str = "resnet34",
    in_channels: int = 3,
    output_dir: str = "predictions",
):
    """
    Kjor prediksjon pa nye bilder.

    Returns: liste med numpy-masker (0/1 per piksel)
    """
    print(f"=== Prediksjon pa {len(image_paths)} bilder ===")

    # Last modell
    model = build_unet(in_channels=in_channels, encoder=encoder)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    predictions = []

    with torch.no_grad():
        for path in image_paths:
            # Last og preprocess
            if path.endswith(".npy"):
                image = np.load(path).astype(np.float32)
            else:
                image = np.array(Image.open(path).convert("RGB").resize(
                    (img_size, img_size)
                )).astype(np.float32) / 255.0

            # Til tensor
            if image.ndim == 2:
                tensor = torch.tensor(image[np.newaxis, np.newaxis, ...])
            else:
                tensor = torch.tensor(image.transpose(2, 0, 1)[np.newaxis, ...])

            # Prediker
            output = model(tensor.to(DEVICE))
            mask = (torch.sigmoid(output) > threshold).cpu().numpy().squeeze()

            predictions.append(mask)

            # Lagre maske som bilde
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(f"{output_dir}/{Path(path).stem}_pred.png")

    print(f"Prediksjoner lagret i {output_dir}/")
    return predictions


# === SUBMISSION-HJELPER ===

def prepare_submission(predictions: list[np.ndarray], image_paths: list[str], output_path: str = "submission.json"):
    """
    Formater prediksjoner for API-submission.

    TODO: Tilpass formatet til oppgavens API-krav.
    Vanlige formater:
    - RLE (Run-Length Encoding) for masker
    - Base64-encodet PNG
    - Numpy arrays som lister
    """
    def mask_to_rle(mask):
        """Konverter binar maske til Run-Length Encoding."""
        pixels = mask.flatten()
        runs = []
        prev = 0
        start = 0
        for i, p in enumerate(pixels):
            if p != prev:
                if prev == 1:
                    runs.append(f"{start} {i - start}")
                start = i
                prev = p
        if prev == 1:
            runs.append(f"{start} {len(pixels) - start}")
        return " ".join(runs) if runs else ""

    submission = []
    for path, mask in zip(image_paths, predictions):
        submission.append({
            "image_id": Path(path).stem,
            "rle_mask": mask_to_rle(mask.astype(np.uint8)),
            # "mask": mask.tolist(),  # Alternativ: hel maske som liste
        })

    with open(output_path, "w") as f:
        json.dump(submission, f)

    print(f"Submission lagret til {output_path} ({len(submission)} prediksjoner)")
    return submission


# === QUICK-START: PRETRAINED MODELL UTEN TRENING ===

def segment_with_pretrained(image_paths: list[str], task: str = "panoptic"):
    """
    Segmentering med pretrained modell — INGEN treningsdata trengs!
    Gir instant baseline.

    task: "panoptic", "semantic", "instance"
    """
    from transformers import pipeline as hf_pipeline

    print(f"=== Pretrained segmentering ({task}) ===")

    segmenter = hf_pipeline("image-segmentation", model="facebook/mask2former-swin-base-coco-panoptic")

    results = []
    for path in image_paths:
        segments = segmenter(path)
        results.append({
            "path": path,
            "segments": [{"label": s["label"], "score": s["score"]} for s in segments],
        })
        labels = [s["label"] for s in segments[:5]]
        print(f"  {Path(path).name}: {labels}")

    return results


# === KJOR ===

if __name__ == "__main__":
    # TODO: Velg riktig modus basert pa oppgaven

    # --- Modus 1: Tren pa egne data ---
    # model = train(
    #     image_dir="data/images",
    #     mask_dir="data/masks",
    #     epochs=50,
    #     batch_size=8,
    #     img_size=256,
    #     encoder="resnet34",  # Prov ogsa: resnet50, efficientnet-b3
    # )

    # --- Modus 2: Prediker med trent modell ---
    # image_files = [str(p) for p in Path("data/test").glob("*.png")]
    # predictions = predict(
    #     model_path="models/segmentation/best_model.pth",
    #     image_paths=image_files,
    #     img_size=256,
    # )
    # prepare_submission(predictions, image_files)

    # --- Modus 3: Pretrained (instant baseline, ingen trening) ---
    # results = segment_with_pretrained(["test_image.jpg"])

    print("Segmentering template klar!")
    print("Uncomment en av modusene over for a starte.")
    print(f"Device: {DEVICE}")
