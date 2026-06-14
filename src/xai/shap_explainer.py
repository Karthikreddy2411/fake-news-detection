"""
src/xai/shap_explainer.py
─────────────────────────────────────────────────────────────────────────────
SHAP explanation for fake-news classifiers (Section VI-E1).

Implementation details from the paper:
  • A TextMasker generates multiple versions of the sentence with hidden words.
  • A classification function f() accepts a list of sentences, converts them
    to token-ID sequences, runs the model, and returns [P(0), P(1)] pairs.
  • 100 partially-masked text variants are generated per instance.
  • Attribution values are assigned to positions in the input sequence.
  • PAD positions are omitted from the explanation.

Output:
  attribution vector of shape (MAX_LEN,) aligned with the token sequence.
  Positive → pushes toward Real (class 1).
  Negative → pushes toward Fake (class 0).
"""

import os
import sys
import numpy as np
import shap
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as C


class SHAPExplainer:
    """
    Wraps shap.Explainer (PartitionExplainer) with a TextMasker
    to produce token-level attributions for the fake-news models.
    """

    def __init__(self, model, tokenizer):
        """
        Parameters
        ----------
        model     : compiled tf.keras.Model (LSTM or CNN)
        tokenizer : fitted Keras Tokenizer
        """
        self.model     = model
        self.tokenizer = tokenizer
        self._index_word = {v: k for k, v in tokenizer.word_index.items()}

        # SHAP explainer is lazy-initialised on first call to save startup time
        self._explainer = None

    # ── Internal classification function ──────────────────────────────────────

    def _predict_fn(self, texts):
        """
        Accepts a list of strings (SHAP masks → words removed / hidden).
        Returns ndarray of shape (N, 2) with [P(fake), P(real)].
        """
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=C.MAX_LEN,
                          padding=C.PADDING, truncating=C.TRUNCATING,
                          value=C.PAD_IDX)
        probs = self.model.predict(X, batch_size=len(texts), verbose=0).ravel()
        return np.column_stack([1 - probs, probs])   # [P(0), P(1)]

    # ── Token-ID sequence → sentence string for the masker ────────────────────

    def _ids_to_text(self, token_ids: np.ndarray) -> str:
        """Reconstruct a readable sentence from a padded integer sequence."""
        words = [
            self._index_word.get(idx, C.OOV_TOKEN)
            for idx in token_ids
            if idx != C.PAD_IDX
        ]
        return " ".join(words)

    # ── Public API ─────────────────────────────────────────────────────────────

    def explain(self, token_ids: np.ndarray, num_samples: int = C.SHAP_NUM_SAMPLES) -> np.ndarray:
        """
        Compute SHAP attributions for a single padded token-ID sequence.

        Parameters
        ----------
        token_ids  : int array of shape (MAX_LEN,)
        num_samples: number of masked variants (100 per paper)

        Returns
        -------
        attributions : float array of shape (MAX_LEN,)
                       (PAD positions have value 0.0)
        """
        text = self._ids_to_text(token_ids)

        masker    = shap.maskers.Text(tokenizer=r"\W+")
        explainer = shap.Explainer(self._predict_fn, masker, output_names=["Fake", "Real"])

        shap_values = explainer(
            [text],
            max_evals=num_samples * 2 + 1,   # controls approximation budget
            batch_size=32,
        )

        # shap_values.values shape: (1, num_tokens, 2)
        # We want class-1 (Real) attributions → index 1
        token_shap  = shap_values.values[0, :, 1]   # shape (num_tokens_in_text,)
        token_names = shap_values.data[0]            # list of word strings

        # Map back to fixed-length position vector
        return self._map_to_positions(token_names, token_shap, token_ids)

    def _map_to_positions(self, shap_tokens, shap_vals, token_ids):
        """
        Align SHAP word-level scores with the padded position sequence.
        This ensures attribution[j] corresponds to token_ids[j].
        """
        attributions = np.zeros(C.MAX_LEN, dtype=np.float32)
        non_pad = [i for i, idx in enumerate(token_ids) if idx != C.PAD_IDX]
        seq_words = [
            self._index_word.get(token_ids[i], C.OOV_TOKEN) for i in non_pad
        ]

        # Greedy word → position alignment
        shap_idx = 0
        for pos, word in zip(non_pad, seq_words):
            if shap_idx < len(shap_tokens):
                attributions[pos] = shap_vals[shap_idx]
                shap_idx += 1

        return attributions

    def explain_text(self, raw_text: str, num_samples: int = C.SHAP_NUM_SAMPLES) -> np.ndarray:
        """Convenience wrapper: takes a raw string, returns attributions."""
        seqs = self.tokenizer.texts_to_sequences([raw_text])
        X = pad_sequences(seqs, maxlen=C.MAX_LEN,
                          padding=C.PADDING, truncating=C.TRUNCATING,
                          value=C.PAD_IDX)
        return self.explain(X[0], num_samples=num_samples)
