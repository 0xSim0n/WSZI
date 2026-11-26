"""Miary jakosci kontrfaktykow i ich odpornosci czasowej.

Odpornosc czasowa kontrfaktyka x' (klasa docelowa c) wzgledem driftu
----------------------------------------------------------------------
Dla koncepcji phi(delta) strumienia (idealny klasyfikator w momencie driftu delta) definiujemy

    R+(x') = sup{ r >= 0 : f(x'; phi(d)) = c  dla wszystkich d in [0, r]  }     (drift "w gore")
    R-(x') = sup{ r >= 0 : f(x'; phi(-d)) = c dla wszystkich d in [0, r]  }     (drift "w dol")
    R(x')  = min(R+, R-)                                                         (najgorszy kierunek)

czyli *zakres przyszlego driftu, przy ktorym przewidywanie docelowe nie ulega zmianie*.
Gdy kontrfaktyk nie jest poprawny juz w delta = 0, R = 0. Przy predkosci driftu v: czas
do uniewaznienia  T = R / v  (liczba krokow strumienia). R liczymy na siatce delta (krok Delta_eval / n).
Dodatkowo: ``survival(delta)`` = odsetek kontrfaktykow poprawnych dla calego odcinka [0, delta].
"""
from __future__ import annotations

import numpy as np

from .streams import Stream


def delta_grid(stream: Stream, factor: float = 2.0, n: int = 81) -> np.ndarray:
    """Siatka delta >= 0 siegajaca factor * Delta (poza granica zalozona w treningu)."""
    return np.linspace(0.0, factor * stream.default_bound, n)


def _valid_matrix(stream: Stream, X_cf: np.ndarray, target: int, signed_deltas: np.ndarray) -> np.ndarray:
    return np.stack([stream.label(X_cf, d) == target for d in signed_deltas], axis=1)  # (n, G)


def robustness_radius(stream: Stream, X_cf: np.ndarray, target: int, deltas: np.ndarray):
    """Zwraca (R_plus, R_minus, S_plus, S_minus): promienie oraz macierze 'przezycia' (n, G)."""
    out = []
    for sign in (+1, -1):
        V = _valid_matrix(stream, X_cf, target, sign * deltas)
        S = np.logical_and.accumulate(V, axis=1)               # poprawny na calym [0, delta]
        n_ok = S.sum(1)
        R = np.where(n_ok > 0, deltas[np.clip(n_ok - 1, 0, None)], 0.0)
        out.append((R, S))
    (Rp, Sp), (Rm, Sm) = out
    return Rp, Rm, Sp, Sm


def proximity(X_q: np.ndarray, X_cf: np.ndarray):
    """Bliskosc: odleglosc L1 i L2 w przestrzeni znormalizowanej [0,1]^d."""
    diff = np.abs(X_cf - X_q)
    return diff.sum(1), np.sqrt((diff ** 2).sum(1))


def sparsity(X_q: np.ndarray, X_cf: np.ndarray, tol: float = 1e-2) -> np.ndarray:
    """Liczba zmienionych cech (|zmiana| > tol)."""
    return (np.abs(X_cf - X_q) > tol).sum(1)


def accuracy_under_drift(clf, stream: Stream, rng: np.random.Generator, n: int = 4000, n_grid: int = 9):
    """Dokladnosc *zamrozonego* klasyfikatora: w delta=0 oraz usredniona po delta in [-Delta, Delta]."""
    acc0 = None
    accs = []
    for d in np.linspace(-stream.default_bound, stream.default_bound, n_grid):
        X, y = stream.sample(n, rng, d)
        a = float((clf.predict(X) == y).mean())
        accs.append(a)
        if abs(d) < 1e-12:
            acc0 = a
    return acc0, float(np.mean(accs))
