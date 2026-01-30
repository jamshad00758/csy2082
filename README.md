# Palm Recognition - Deep Learning Classifier

This project implements a multi-class palm recognition system using a custom dataset of close-up palm images. It includes dataset preparation, preprocessing, model training, evaluation (including unknown-palm rejection), and live camera testing.

## Project structure

- `data/raw/` - raw images grouped by identity (`data/raw/<person_id>/*.jpg`)
- `data/processed/` - preprocessed images created by the toolkit
- `data/unknown/` - optional images of unknown palms for open-set testing
- `models/` - trained model checkpoints
- `reports/` - evaluation outputs (metrics, confusion matrix)
- `src/palm_recognition.py` - single script for capture, preprocess, train, evaluate, webcam

## Dataset design (feature choice)

This solution uses palm texture and crease patterns from **close-up grayscale images**. The preprocessing pipeline enhances contrast (CLAHE) and standardizes size to emphasize line and texture structure rather than color.

Recommended capture guidance:

- Keep the palm centered, fingers out of frame, consistent distance.
- Use even lighting (avoid harsh shadows).
- Capture 25–40 images per person from slightly different angles.

## Setup

Create a Python environment and install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 1) Preprocess dataset

If you do not have a dataset yet, you can generate a small synthetic sample set:

```
python src/palm_recognition.py download --output data/raw --classes 4 --images-per-class 30
```

Or start by capturing your own images:

```
python src/palm_recognition.py capture --output data/raw/person_01
```

After capturing for each person, preprocess:

```
python src/palm_recognition.py preprocess --input data/raw --output data/processed
```

This applies grayscale conversion, histogram equalization (CLAHE), denoising, and resizing.

## 2) Train model

```
python src/palm_recognition.py train --data data/processed --epochs 30 --batch-size 32
```

Outputs:

- `models/palm_cnn.pt` (best checkpoint)
- `models/classes.json` (label mapping)
- `reports/train_history.csv`

## 3) Evaluate model

```
python src/palm_recognition.py evaluate --data data/processed --model models/palm_cnn.pt
```

Optional open-set testing with unknown palms:

```
python src/palm_recognition.py evaluate --data data/processed --model models/palm_cnn.pt --unknown data/unknown
```

Outputs:

- accuracy, macro-F1, confusion matrix
- unknown-palm rejection rate (if unknown set provided)
- `reports/confusion_matrix.png`

## 4) Live camera test

```
python src/palm_recognition.py webcam --model models/palm_cnn.pt --classes models/classes.json
```

The system displays the predicted identity with a confidence score. If the confidence is below the threshold, it reports **Unknown**.

## Metrics rationale

- **Accuracy** provides a baseline for identification.
- **Macro F1** is chosen to handle class imbalance.
- **Confusion matrix** helps diagnose specific mis-identifications.
- **Unknown rejection rate** evaluates robustness to unseen palms.

## Bias and ethical considerations

- **Sample bias**: A small dataset can overfit to specific lighting, skin tone, or capture device. Mitigation: collect diverse samples per user and normalize via preprocessing.
- **Privacy**: Palm images are biometric data. Store locally, anonymize identities (IDs like `person_01`), and avoid sharing raw images.
- **Consent**: Ensure explicit permission from other participants.
- **Security risk**: This is a prototype; do not deploy for high-stakes authentication without extensive testing.

## Suggested report outline

- Dataset design and capture protocol
- Preprocessing pipeline and justification
- Model architecture and hyperparameters
- Evaluation metrics and results
- Open-set handling (unknown palm threshold)
- Bias/ethics analysis

