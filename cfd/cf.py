"""Generowanie kontrfaktykow metoda DiCE (pakiet ``dice-ml``, wariant gradientowy dla PyTorch).

DiCE jest metoda wyjsciowa projektu. Uzywamy jej *bez modyfikacji* - roznica miedzy porownywanymi
podejsciami lezy wylacznie w klasyfikatorze (jaka funkcja prawdopodobienstwa jest podawana DiCE)
oraz w progu ``stopping_threshold`` (min. P(klasa docelowa) wymagane od kontrfaktyka).
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from .models import Classifier
from .streams import Stream

warnings.filterwarnings("ignore")
logging.getLogger("dice_ml").setLevel(logging.ERROR)

import dice_ml  # noqa: E402


def _background_df(stream: Stream, X: np.ndarray, y: np.ndarray, n: int = 500) -> pd.DataFrame:
    """Dane referencyjne dla DiCE (zakresy cech, MAD). Wiersze zakotwiczajace 0/1 gwarantuja, ze
    normalizacja min-max w DiCE jest dokladnie identycznoscia (modele dzialaja w [0,1]^d)."""
    df = pd.DataFrame(X[:n], columns=stream.feature_names)
    df["y"] = y[:n]
    lo = pd.DataFrame([np.zeros(stream.d), np.ones(stream.d)], columns=stream.feature_names)
    lo["y"] = [0, 1]
    return pd.concat([df, lo], ignore_index=True)


class DiceGenerator:
    """Opakowanie DiCE: dla (klasyfikator, klasa docelowa) tworzy eksplainer i generuje kontrfaktyki."""

    def __init__(self, clf: Classifier, stream: Stream, X_bg: np.ndarray, y_bg: np.ndarray,
                 tau: float = 0.5, max_iter: int = 1500, learning_rate: float = 0.01,
                 proximity_weight: float = 0.5, min_iter: int = 300, loss_diff_thres: float = 1e-6,
                 loss_converge_maxiter: int = 3, seed: int = 0):
        self.clf, self.stream, self.tau, self.seed = clf, stream, tau, seed
        self.kw = dict(max_iter=max_iter, learning_rate=learning_rate, proximity_weight=proximity_weight,
                       min_iter=min(min_iter, max_iter), loss_diff_thres=loss_diff_thres,
                       loss_converge_maxiter=loss_converge_maxiter, posthoc_sparsity_algorithm="binary")
        self.data = dice_ml.Data(dataframe=_background_df(stream, X_bg, y_bg),
                                 continuous_features=list(stream.feature_names), outcome_name="y")
        self._explainers = {}

    def _explainer(self, target: int):
        if target not in self._explainers:
            model = dice_ml.Model(model=self.clf.dice_module(target), backend="PYT")
            self._explainers[target] = dice_ml.Dice(self.data, model, method="gradient")
        return self._explainers[target]

    def generate(self, X_query: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
        """Zwraca (X_cf, ok): po jednym kontrfaktyku na zapytanie; ok=False gdy DiCE nic nie znalazl."""
        np.random.seed(self.seed)
        import torch
        torch.manual_seed(self.seed)
        exp = self._explainer(target)
        names = self.stream.feature_names
        X_cf = np.array(X_query, dtype=float).copy()
        ok = np.zeros(len(X_query), dtype=bool)
        for i, x in enumerate(X_query):
            q = pd.DataFrame([x], columns=names)
            try:
                res = exp.generate_counterfactuals(
                    q, total_CFs=1, desired_class=1, stopping_threshold=self.tau,
                    features_to_vary="all", verbose=False, **self.kw)
                cf_df = res.cf_examples_list[0].final_cfs_df
                if cf_df is not None and len(cf_df):
                    X_cf[i] = cf_df[names].to_numpy(dtype=float)[0]
                    ok[i] = True
            except Exception:  # DiCE potrafi rzucic wyjatek, gdy nie znajdzie rozwiazania
                pass
        return np.clip(X_cf, 0.0, 1.0), ok
