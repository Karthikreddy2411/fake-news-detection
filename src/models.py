"""
src/models.py
─────────────────────────────────────────────────────────────────────────────
LSTM and CNN model architectures as described in Section VI-C.

LSTM:
  Embedding(VOCAB_SIZE, 128, input_length=750)
  → LSTM(128, dropout=0.2, recurrent_dropout=0.2)
  → Dense(1, activation='sigmoid')

CNN:
  Embedding(VOCAB_SIZE, 128, input_length=750)
  → Conv1D(128, kernel_size=5, activation='relu')
  → GlobalAveragePooling1D()
  → Dropout(0.5)
  → Dense(1, activation='sigmoid')

Both compiled with: Adam / binary_crossentropy / accuracy
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as C

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    Conv1D, GlobalAveragePooling1D,
)


def build_lstm_model() -> tf.keras.Model:
    """
    Sequential LSTM model (Section VI-C, paragraph a).
    Captures long-range sequential dependencies via gating.
    """
    model = Sequential(name="LSTM_FakeNews", layers=[
        Embedding(
            input_dim=C.VOCAB_SIZE,
            output_dim=C.EMBED_DIM,
            input_length=C.MAX_LEN,
            name="embedding",
        ),
        LSTM(
            units=C.LSTM_UNITS,
            dropout=C.LSTM_DROPOUT,
            # NOTE: recurrent_dropout removed — it disables the fast Metal/cuDNN
            # kernel on Apple Silicon, causing 10-20x slower training on CPU.
            # Regularisation is preserved via the Dropout layer below.
            # unroll=True prevents TF threading deadlock on macOS.
            unroll=True,
            name="lstm",
        ),
        Dropout(C.LSTM_DROPOUT, name="dropout"),
        Dense(1, activation="sigmoid", name="output"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_model() -> tf.keras.Model:
    """
    1-D Convolutional model (Section VI-C, paragraph b).
    Detects local n-gram patterns; uses GlobalAveragePooling for XAI stability.
    """
    model = Sequential(name="CNN_FakeNews", layers=[
        Embedding(
            input_dim=C.VOCAB_SIZE,
            output_dim=C.EMBED_DIM,
            input_length=C.MAX_LEN,
            name="embedding",
        ),
        Conv1D(
            filters=C.CNN_FILTERS,
            kernel_size=C.CNN_KERNEL_SIZE,
            activation="relu",
            name="conv1d",
        ),
        GlobalAveragePooling1D(name="global_avg_pool"),
        Dropout(C.CNN_DROPOUT, name="dropout"),
        Dense(1, activation="sigmoid", name="output"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model(path: str) -> tf.keras.Model:
    """Load a saved .keras model from disk."""
    return tf.keras.models.load_model(path)


def summarise(model: tf.keras.Model):
    """Print a clean model summary."""
    model.summary(line_length=80)


if __name__ == "__main__":
    print("=== LSTM ===")
    summarise(build_lstm_model())
    print("\n=== CNN ===")
    summarise(build_cnn_model())
