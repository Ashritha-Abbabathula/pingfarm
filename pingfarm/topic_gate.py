"""A learned "can our knowledge base answer this?" check.

The first version of the gate only looked at how close the best passage was. That separates real questions from
off-topic ones only moderately (AUC 0.84) and fails on NEAR MISSES: a farmer asking about tomato fertiliser or
cotton pests uses words that look close to our tomato documents.

This version is a tiny logistic regression on the SAME multilingual question embeddings. It is trained on real
Kisan Call Centre questions: tomato/potato questions our documents can answer (positives) versus off-topic
questions, other crops and tomato/potato questions on topics we do not cover (negatives).
Because the embeddings are multilingual, training on English transfers to Hindi and Telugu (measured in
scripts/train_topic_gate.py).
"""
import json
import re
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "topic_gate.json"


class TopicGate:
    def __init__(self, weights: np.ndarray, bias: float, threshold: float, threshold_indic: float | None = None):
        self.w, self.b, self.threshold = weights, bias, threshold
        self.threshold_indic = threshold_indic if threshold_indic is not None else threshold

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "TopicGate | None":
        if not Path(path).exists():
            return None
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["w"], dtype="float32"), float(d["b"]), float(d["threshold"]), d.get("threshold_indic"))

    def save(self, path: Path = DEFAULT_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"w": self.w.tolist(), "b": self.b, "threshold": self.threshold,
                                        "threshold_indic": self.threshold_indic}))

    def score(self, query_vector: np.ndarray) -> float:
        """Probability that the knowledge base can answer this question."""
        z = float(query_vector @ self.w + self.b)
        return 1.0 / (1.0 + np.exp(-z))

    def answerable(self, query_vector: np.ndarray, text: str = "") -> bool:
        """Hindi and Telugu questions use their own cut-off (their scores sit lower than English ones)."""
        indic = bool(re.search(r"[ऀ-ॿఀ-౿]", text))
        return self.score(query_vector) >= (self.threshold_indic if indic else self.threshold)
