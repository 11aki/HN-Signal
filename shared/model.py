"""
HNModel — the ML model, shared between trainer (fit) and predictor (predict_proba).

Keeping it in shared/ means both sides use identical inference logic —
training and serving can never drift apart.

How it works:
  1. Title → sentence-transformer → 384-dim embedding vector
     (captures semantic meaning: "I built a Rust compiler" ≈ "Show HN: new systems lang")
  2. Tabular features (score, hour, keywords, etc.) → numpy array
  3. Concatenate both → feed into LogisticRegression
  4. Output: probability 0.0–1.0 that the story will blow up
"""
import pickle
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# fastembed runs all-MiniLM-L6-v2 via ONNX Runtime (~25MB) instead of PyTorch (~700MB).
# Same model weights, same 384-dim embeddings, fraction of the size.
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HNModel:
    """
    Combines sentence-transformer title embeddings with tabular features.

    Why combine both?
    - Tabular features (score, time, keywords) are fast and interpretable
    - Title embeddings capture semantic meaning that keywords miss
    - Together they're stronger than either alone
    """

    def __init__(self, transformer_name: str = SENTENCE_MODEL, C: float = 1.0):
        self.transformer_name = transformer_name
        # C controls regularisation strength in logistic regression.
        # Lower C = simpler model (less overfitting). Default 1.0 is a safe start.
        self.C = C
        self.encoder: TextEmbedding | None = None  # set during fit()
        self.clf: Pipeline | None = None           # set during fit()

    def fit(self, X_tab: np.ndarray, titles: list[str], y: np.ndarray):
        """
        Train the model.

        X_tab: tabular feature matrix, shape (N, num_features)
        titles: list of N story titles
        y: binary labels, 1 = blew up, 0 = didn't
        """
        self.encoder = TextEmbedding(self.transformer_name)
        X_emb = np.array(list(self.encoder.embed(titles)))

        # Stack tabular features and embeddings side-by-side into one matrix
        X = np.hstack([X_tab, X_emb])

        # Pipeline: scale all features to similar ranges, then fit logistic regression.
        # class_weight="balanced" compensates for class imbalance — very few stories
        # actually blow up, so without this the model would just predict "no" for everything.
        self.clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=1000, class_weight="balanced")),
        ])
        self.clf.fit(X, y)

    def predict_proba(self, X_tab: np.ndarray, titles: list[str]) -> np.ndarray:
        """
        Return blow-up probability for each story (values between 0.0 and 1.0).
        """
        X_emb = np.array(list(self.encoder.embed(titles)))
        X = np.hstack([X_tab, X_emb])
        # predict_proba returns [[prob_negative, prob_positive], ...]
        # We only want the positive (blow-up) probability, hence [:, 1]
        return self.clf.predict_proba(X)[:, 1]

    def __getstate__(self):
        # ONNX InferenceSession is not picklable — exclude it.
        # The encoder is re-initialized from the baked-in model cache on load.
        state = self.__dict__.copy()
        state["encoder"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.transformer_name:
            self.encoder = TextEmbedding(self.transformer_name)


def save_artifact(model: "HNModel", version: str, path: Path):
    """
    Save model + version string to a .pkl file using pickle.

    The version string (e.g. "v20240402-120000") is stored alongside
    the model so the predictor can log which model version made each prediction.
    """
    with open(path, "wb") as f:
        pickle.dump({"model": model, "version": version}, f)


def load_artifact(path: Path) -> tuple["HNModel", str]:
    """Load a saved model artifact. Returns (model, version_string)."""
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["version"]
