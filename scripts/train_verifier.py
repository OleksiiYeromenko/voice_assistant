#!/usr/bin/env python3
"""Train a custom verifier model for the wake word detector.

Uses labeled captures from data/wake_captures/true_positives/ and
data/wake_captures/false_positives/ to train a LogisticRegression classifier
on openWakeWord's internal AudioSet embeddings.

Usage:
    uv run scripts/train_verifier.py
"""

from pathlib import Path

from openwakeword.custom_verifier_model import train_custom_verifier

MODEL = "./models/hey_Poon-dyk_acc_0.84_recall_0.69_fp_4.5.onnx"
TP_DIR = Path("data/wake_captures/true_positives")
FP_DIR = Path("data/wake_captures/false_positives")
OUT = "models/verifier.pkl"

tp = sorted(TP_DIR.glob("*.wav"))
fp = sorted(FP_DIR.glob("*.wav"))

if not tp:
    raise SystemExit(f"No positive examples found in {TP_DIR}")
if not fp:
    raise SystemExit(f"No negative examples found in {FP_DIR}")

print(f"Training verifier: {len(tp)} positive, {len(fp)} negative examples")
train_custom_verifier(tp, fp, output_path=OUT, model_name=MODEL)
print(f"Saved → {OUT}")
