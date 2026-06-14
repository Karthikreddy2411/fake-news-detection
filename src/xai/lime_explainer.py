"""
src/xai/lime_explainer.py
─────────────────────────────────────────────────────────────────────────────
LIME explanation for fake-news classifiers (Section VI-E2).

Implementation details from the paper:
  • The classifier function converts text → token IDs → model → class probabilities.
  • LIME creates ~1000 perturbed versions of the input (word deletions/replacements).
  • A simple linear surrogate is trained on the local neighbourhood.
  • Up to 20 of the most influential words are surfaced.
  • Word weights are mapped back to positions in the padded input sequence.

Output:
  attribution vector of shape (MAX_LEN,) aligned with the token sequence.
  Positive → Real; Negative → Fake.
"""

import os
import sys
import numpy as np
from lime.lime_text import LimeTextExplainer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as C


class LIMEExplainer:
    """
    Wraps lime.lime_text.LimeTextExplainer to produce
    position-aligned attribution vectors for the fake-news models.
    """

    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self._index_word = {v: k for k, v in tokenizer.word_index.items()}

        self._lime = LimeTextExplainer(
            class_names=["Fake", "Real"],
            random_state=C.RANDOM_STATE,
        )

    # ── Classification function ────────────────────────────────────────────────

    def _predict_fn(self, texts):
        """
        Accepts a list of strings (LIME perturbations).
        Returns ndarray of shape (N, 2): [P(fake), P(real)].
        """
        seqs = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=C.MAX_LEN,
                          padding=C.PADDING, truncating=C.TRUNCATING,
                          value=C.PAD_IDX)
        probs = self.model.predict(X, batch_size=max(1, len(texts)), verbose=0).ravel()
        return np.column_stack([1 - probs, probs])

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ids_to_text(self, token_ids: np.ndarray) -> str:
        return " ".join(
            self._index_word.get(idx, C.OOV_TOKEN)
            for idx in token_ids
            if idx != C.PAD_IDX
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def explain(
        self,
        token_ids: np.ndarray,
        num_samples: int = C.LIME_NUM_SAMPLES,
        num_features: int = C.LIME_NUM_FEATURES,
    ) -> np.ndarray:
        """
        Compute LIME attributions for a single padded token-ID sequence.

        Parameters
        ----------
        token_ids   : int array of shape (MAX_LEN,)
        num_samples : perturbation budget (1000 per paper)
        num_features: top-N words to expose (20 per paper)

        Returns
        -------
        attributions : float array of shape (MAX_LEN,)
        """
        text = self._ids_to_text(token_ids)

        exp = self._lime.explain_instance(
            text,
            self._predict_fn,
            num_samples=num_samples,
            num_features=num_features,
            labels=(1,),   # explain class 1 (Real)
        )

        # exp.as_list(label=1) → list of (word, weight) tuples
        word_weights = dict(exp.as_list(label=1))

        return self._map_to_positions(word_weights, token_ids)

    def _map_to_positions(self, word_weights: dict, token_ids: np.ndarray) -> np.ndarray:
        """
        Map {word: weight} → position vector of length MAX_LEN.
        Each position j gets the LIME weight of the word at token_ids[j].
        """
        attributions = np.zeros(C.MAX_LEN, dtype=np.float32)
        for pos, idx in enumerate(token_ids):
            if idx == C.PAD_IDX:
                continue
            word = self._index_word.get(idx, "")
            if word in word_weights:
                attributions[pos] = word_weights[word]
        return attributions

    def explain_text(self, raw_text: str, **kwargs) -> np.ndarray:
        """Convenience wrapper: raw string → attributions."""
        seqs = self.tokenizer.texts_to_sequences([raw_text])
        X = pad_sequences(seqs, maxlen=C.MAX_LEN,
                          padding=C.PADDING, truncating=C.TRUNCATING,
                          value=C.PAD_IDX)
        return self.explain(X[0], **kwargs)
