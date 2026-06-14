"""
src/xai/ig_explainer.py
─────────────────────────────────────────────────────────────────────────────
Integrated Gradients (IG) for fake-news classifiers (Section VI-E3).

Implementation details from the paper:
  • The model is split into two parts: the Embedding layer and the rest.
  • Baseline = sequence of all PADs → embedding b_emb
  • 50 points are sampled along the straight line from b_emb to x_emb.
  • For each interpolation point the gradient of the output w.r.t. embeddings
    is computed using tf.GradientTape.
  • Averaged gradients are multiplied by (x_emb − b_emb).
  • The result is summed over the embedding dimensions → vector of length L.
  • PAD positions contribute ≈ 0 (baseline is PAD).

Output:
  attribution vector of shape (MAX_LEN,) aligned with token positions.
  Positive → Real; Negative → Fake.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config as C


class IGExplainer:
    """
    Integrated Gradients explainer.
    Works directly in the embedding space — no text conversion required.
    Particularly well-suited for CNN architectures (Section VII).
    """

    def __init__(self, model, tokenizer):
        self.model     = model
        self.tokenizer = tokenizer
        self._index_word = {v: k for k, v in tokenizer.word_index.items()}

        # Split model into embedding sub-model and the rest
        self._embedding_layer = model.get_layer("embedding")
        self._rest_model       = self._build_rest_model(model)

    # ── Model splitting ────────────────────────────────────────────────────────

    @staticmethod
    def _build_rest_model(model: tf.keras.Model) -> tf.keras.Model:
        """
        Build a functional sub-model that takes embedding outputs as input
        and produces the final scalar prediction.
        This lets us differentiate through everything after the embedding.
        """
        # Get embedding layer output shape
        emb_layer  = model.get_layer("embedding")
        emb_output = emb_layer.output  # shape (None, MAX_LEN, EMBED_DIM)

        # Trace the rest of the graph from the embedding output
        rest_input = tf.keras.Input(shape=emb_output.shape[1:], name="emb_input")
        x = rest_input
        found_emb = False
        for layer in model.layers:
            if layer.name == "embedding":
                found_emb = True
                continue
            if found_emb:
                x = layer(x)

        return tf.keras.Model(inputs=rest_input, outputs=x, name="rest_model")

    # ── Integrated Gradients core ──────────────────────────────────────────────

    def _integrated_gradients(
        self,
        x_ids: np.ndarray,         # shape (MAX_LEN,)
        baseline_ids: np.ndarray,  # shape (MAX_LEN,) — all zeros (PAD)
        steps: int = C.IG_STEPS,
    ) -> np.ndarray:
        """
        Compute IG attributions in embedding space.
        Returns float array of shape (MAX_LEN,).
        """
        # Compute embedding vectors
        x_ids_t   = tf.constant([x_ids],        dtype=tf.int32)
        b_ids_t   = tf.constant([baseline_ids], dtype=tf.int32)

        x_emb = self._embedding_layer(x_ids_t)[0]   # (MAX_LEN, EMBED_DIM)
        b_emb = self._embedding_layer(b_ids_t)[0]   # (MAX_LEN, EMBED_DIM)

        delta = x_emb - b_emb  # (MAX_LEN, EMBED_DIM)

        # Interpolate between baseline and input
        alphas = tf.linspace(0.0, 1.0, steps + 1)   # (steps+1,)

        # Accumulate gradients
        grad_sum = tf.zeros_like(x_emb)              # (MAX_LEN, EMBED_DIM)

        for alpha in alphas:
            interp = b_emb + alpha * delta           # (MAX_LEN, EMBED_DIM)
            interp_batched = tf.expand_dims(interp, 0)  # (1, MAX_LEN, EMBED_DIM)

            with tf.GradientTape() as tape:
                tape.watch(interp_batched)
                pred = self._rest_model(interp_batched, training=False)  # (1,1)

            grads = tape.gradient(pred, interp_batched)  # (1, MAX_LEN, EMBED_DIM)
            grad_sum = grad_sum + grads[0]

        # Average and multiply by (x_emb - b_emb)
        avg_grads    = grad_sum / (steps + 1)        # (MAX_LEN, EMBED_DIM)
        attributions = avg_grads * delta             # (MAX_LEN, EMBED_DIM)

        # Sum over embedding dimensions → (MAX_LEN,)
        return tf.reduce_sum(attributions, axis=-1).numpy()

    # ── Public API ─────────────────────────────────────────────────────────────

    def explain(self, token_ids: np.ndarray, steps: int = C.IG_STEPS) -> np.ndarray:
        """
        Compute IG attributions for a single padded token-ID sequence.

        Parameters
        ----------
        token_ids : int array of shape (MAX_LEN,)
        steps     : number of integration steps (32–50 per paper)

        Returns
        -------
        attributions : float array of shape (MAX_LEN,)
                       PAD positions ≈ 0 (baseline is PAD).
        """
        baseline = np.zeros(C.MAX_LEN, dtype=np.int32)  # all-PAD sequence
        return self._integrated_gradients(token_ids, baseline, steps=steps)

    def explain_text(self, raw_text: str, steps: int = C.IG_STEPS) -> np.ndarray:
        """Convenience wrapper: raw string → attributions."""
        seqs = self.tokenizer.texts_to_sequences([raw_text])
        X = pad_sequences(seqs, maxlen=C.MAX_LEN,
                          padding=C.PADDING, truncating=C.TRUNCATING,
                          value=C.PAD_IDX)
        return self.explain(X[0], steps=steps)
