"""
src/train.py
─────────────────────────────────────────────────────────────────────────────
Training and evaluation of both models (Section VI-D).

Training setup (paper):
  - Loss:       binary_crossentropy
  - Optimizer:  Adam (default lr=1e-3)
  - Metric:     accuracy
  - Batch size: 512 (large for CPU parallelism)
  - Epochs:     5 with early stopping

Saves:
  models/lstm_model.keras
  models/cnn_model.keras
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config as C

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models import build_lstm_model, build_cnn_model


def train_model(
    model: tf.keras.Model,
    X_train, y_train,
    X_val,   y_val,
    save_path: str,
) -> tf.keras.callbacks.History:
    """
    Train a compiled Keras model, save the best checkpoint.
    Returns training history.
    """
    os.makedirs(C.MODELS_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=C.EPOCHS,
        batch_size=C.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_model(model: tf.keras.Model, X_test, y_test, name: str = ""):
    """
    Print accuracy + full classification report on the test set.
    Returns dict with accuracy, precision, recall, f1.
    """
    print(f"\n{'─'*50}")
    print(f"  Evaluation — {name}")
    print(f"{'─'*50}")

    loss, acc = model.evaluate(X_test, y_test, batch_size=C.BATCH_SIZE, verbose=0)
    y_pred = (model.predict(X_test, batch_size=C.BATCH_SIZE, verbose=0) >= 0.5).astype(int).ravel()

    print(f"  Loss     : {loss:.4f}")
    print(f"  Accuracy : {acc:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix:\n{cm}\n")

    from sklearn.metrics import precision_score, recall_score, f1_score
    return {
        "accuracy":  acc,
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
    }


def train_all(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train both LSTM and CNN, evaluate on the test set.
    Returns both trained models.
    """
    print("\n" + "="*60)
    print("  Training LSTM model")
    print("="*60)
    lstm = build_lstm_model()
    lstm.summary(line_length=70)
    train_model(lstm, X_train, y_train, X_val, y_val, C.LSTM_MODEL_PATH)
    lstm_stats = evaluate_model(lstm, X_test, y_test, "LSTM")

    print("\n" + "="*60)
    print("  Training CNN model")
    print("="*60)
    cnn = build_cnn_model()
    cnn.summary(line_length=70)
    train_model(cnn, X_train, y_train, X_val, y_val, C.CNN_MODEL_PATH)
    cnn_stats = evaluate_model(cnn, X_test, y_test, "CNN")

    print("\n  ✓ Models saved:")
    print(f"    {C.LSTM_MODEL_PATH}")
    print(f"    {C.CNN_MODEL_PATH}")

    return lstm, cnn, lstm_stats, cnn_stats


if __name__ == "__main__":
    from src.preprocess import prepare_data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), tok, df_test = prepare_data()
    train_all(X_train, y_train, X_val, y_val, X_test, y_test)
