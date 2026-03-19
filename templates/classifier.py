"""
NM i AI 2026 — Klassifisering Template
========================================
Rask tekst- og bilde-klassifisering med transformers og scikit-learn.

Slik fungerer klassifisering:
- Du har data med ETIKETTER (f.eks. "spam"/"ikke spam")
- Modellen laerer a forutsi riktig etikett for ny data
- Evalueres med accuracy, F1-score, etc.

Bruk:
1. Velg tekst- eller bilde-klassifisering
2. Tilpass data-lasting
3. Kjor: python classifier.py

TODO-markerte steder ma tilpasses for den spesifikke oppgaven.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional

# scikit-learn for klassiske metoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Transformers for state-of-the-art
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


# === DATALASTING ===

def load_text_data(data_path: str) -> tuple[list[str], list[str]]:
    """
    Last inn tekstdata for klassifisering.

    TODO: Tilpass dette til dataformatet i oppgaven.
    Forventet format: JSON-liste med {"text": "...", "label": "..."}
    """
    data_file = Path(data_path)

    if data_file.suffix == ".json":
        with open(data_file) as f:
            data = json.load(f)
        texts = [item["text"] for item in data]   # TODO: Tilpass feltnavn
        labels = [item["label"] for item in data]  # TODO: Tilpass feltnavn

    elif data_file.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(data_file)
        texts = df["text"].tolist()    # TODO: Tilpass kolonnenavn
        labels = df["label"].tolist()  # TODO: Tilpass kolonnenavn

    else:
        raise ValueError(f"Ustottet filformat: {data_file.suffix}")

    print(f"Lastet {len(texts)} eksempler med {len(set(labels))} klasser")
    print(f"Klasser: {sorted(set(labels))}")
    return texts, labels


def load_image_data(data_path: str) -> tuple[list[str], list[str]]:
    """
    Last inn bildedata for klassifisering.

    TODO: Tilpass dette til dataformatet.
    Forventet: mappe med undermapper per klasse (klasse1/bilde1.jpg, ...)
    """
    data_dir = Path(data_path)
    image_paths = []
    labels = []

    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir():
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    image_paths.append(str(img_path))
                    labels.append(class_dir.name)

    print(f"Lastet {len(image_paths)} bilder med {len(set(labels))} klasser")
    return image_paths, labels


# === METODE 1: EMBEDDING + SKLEARN (Rask og enkel) ===

def classify_with_embeddings(texts: list[str], labels: list[str], test_texts: Optional[list[str]] = None):
    """
    Rask klassifisering:
    1. Konverter tekst til embeddings med sentence-transformers
    2. Tren en sklearn-klassifikator pa embeddingene

    Fordeler: Raskt a trene, fungerer med lite data
    Ulemper: Ikke like bra som fine-tuning pa store datasett
    """
    print("\n=== Embedding + sklearn klassifisering ===")

    # Generer embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Genererer embeddings...")
    X = model.encode(texts, show_progress_bar=True)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Prov flere klassifikatorer
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    }

    best_clf = None
    best_score = 0

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f"  {name}: accuracy = {score:.4f}")

        if score > best_score:
            best_score = score
            best_clf = clf

    # Detaljert rapport for beste modell
    y_pred = best_clf.predict(X_test)
    print(f"\n  Beste modell: {type(best_clf).__name__} ({best_score:.4f})")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Klassifiser nye tekster hvis gitt
    if test_texts:
        test_X = model.encode(test_texts)
        predictions = best_clf.predict(test_X)
        predicted_labels = le.inverse_transform(predictions)

        print("\nPrediksjoner for nye tekster:")
        for text, label in zip(test_texts, predicted_labels):
            print(f"  '{text[:50]}...' -> {label}")

        return predicted_labels

    return best_clf, le, model


# === METODE 2: ZERO-SHOT KLASSIFISERING (Ingen trening!) ===

def classify_zero_shot(texts: list[str], candidate_labels: list[str]) -> list[dict]:
    """
    Zero-shot klassifisering — INGEN treningsdata trengs!

    Modellen klassifiserer tekst til kategorier den aldri har sett for.
    Perfekt for raskt a fa baseline nol oppgaven gis.

    TODO: Juster candidate_labels til oppgaven.
    """
    print("\n=== Zero-shot klassifisering ===")
    print(f"Kategorier: {candidate_labels}")

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",  # Flersproklig: "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    )

    results = []
    for i, text in enumerate(texts):
        result = classifier(text, candidate_labels)
        best_label = result["labels"][0]
        best_score = result["scores"][0]

        results.append({
            "text": text,
            "label": best_label,
            "confidence": best_score,
            "all_scores": dict(zip(result["labels"], result["scores"])),
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(texts)}] '{text[:40]}...' -> {best_label} ({best_score:.3f})")

    return results


# === METODE 3: BILDE-KLASSIFISERING ===

def classify_images(image_paths: list[str], candidate_labels: Optional[list[str]] = None):
    """
    Bilde-klassifisering med pre-trent modell.

    Uten labels: bruker ImageNet-klasser
    Med labels: zero-shot med CLIP
    """
    print("\n=== Bilde-klassifisering ===")

    if candidate_labels:
        # Zero-shot bilde-klassifisering med CLIP
        classifier = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
        )

        results = []
        for path in image_paths:
            result = classifier(path, candidate_labels=candidate_labels)
            best = result[0]
            results.append({"path": path, "label": best["label"], "score": best["score"]})
            print(f"  {Path(path).name} -> {best['label']} ({best['score']:.3f})")

        return results
    else:
        # Standard bilde-klassifisering
        classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

        results = []
        for path in image_paths:
            result = classifier(path)
            best = result[0]
            results.append({"path": path, "label": best["label"], "score": best["score"]})
            print(f"  {Path(path).name} -> {best['label']} ({best['score']:.3f})")

        return results


# === SUBMISSION-HJELPER ===

def format_predictions(predictions: list[dict], output_path: str = "predictions.json"):
    """
    Formater prediksjoner for API-submission.

    TODO: Tilpass output-formatet til oppgavens API-krav.
    """
    output = []
    for pred in predictions:
        output.append({
            "id": pred.get("id", ""),
            "label": pred["label"],
            "confidence": pred.get("confidence", 1.0),
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Prediksjoner lagret til {output_path}")
    return output


# === KJOR ===

if __name__ == "__main__":
    # TODO: Velg riktig metode basert pa oppgaven

    # --- Eksempel 1: Zero-shot (ingen treningsdata trengs) ---
    test_texts = [
        "Pasienten har vondt i hodet og kvalme",
        "Aksjekursen steg med 5% i dag",
        "Liverpool vant 3-1 mot Manchester United",
    ]
    candidate_labels = ["helse", "okonomi", "sport", "teknologi"]

    results = classify_zero_shot(test_texts, candidate_labels)
    for r in results:
        print(f"  {r['label']} ({r['confidence']:.2f}): {r['text'][:50]}")

    # --- Eksempel 2: Embedding + sklearn (trenger treningsdata) ---
    # texts, labels = load_text_data("data/train.json")
    # classify_with_embeddings(texts, labels, test_texts=["ny tekst a klassifisere"])

    # --- Eksempel 3: Bilde-klassifisering ---
    # results = classify_images(["bilde1.jpg", "bilde2.jpg"], candidate_labels=["katt", "hund"])
