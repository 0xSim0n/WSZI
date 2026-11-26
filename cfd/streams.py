"""Strumienie SEA i Hyperplane z kontrolowanym, formalnie ograniczonym driftem.

Formalizacja driftu
-------------------
Koncept strumienia opisuje parametr skalarny ``delta`` (znakowany):

    y = f(x; phi(delta)),     phi(0) = koncept poczatkowy.

* SEA:        phi(delta) = theta0 + delta          (prog theta w jednostkach oryginalnych,
                                                    x_i in [0, 10]; klasa 1 <=> x1 + x2 <= theta)
* Hyperplane: phi(delta) = w0 + delta * s           (s_i in {-1, 0, +1}: wzorzec driftu wag;
                                                    klasa 1 <=> w.x >= 0.5 * sum(w))

Drift jest *ograniczony*: |delta(t)| <= Delta (``default_bound``), a predkosc zmiany wynosi
|d delta / dt| <= v. Zatem po t krokach strumien jest w koncepcie delta(t) = dir * min(v*t, Delta).
Wielkosc driftu mozna tez wyrazic miara niezalezna od parametryzacji - odsetkiem punktow,
ktorym zmienia sie etykieta: ``disagreement(delta) = P_x[ f(x; 0) != f(x; delta) ]``.

Wszystkie cechy sa zwracane w kostce jednostkowej [0, 1]^d (SEA jest przeskalowane /10),
dzieki czemu ta sama siec / ten sam DiCE (normalizacja min-max = identycznosc) obsluguje
oba strumienie.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Stream:
    name: str
    d: int
    default_bound: float          # Delta - maksymalny dopuszczalny |delta|
    speed: float                  # v - maksymalna zmiana delta na krok
    delta_label: str              # opis osi driftu do wykresow
    feature_names: list = field(default_factory=list)

    # --- do nadpisania ---------------------------------------------------
    def label(self, X: np.ndarray, delta: float = 0.0) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    # --- wspolne ---------------------------------------------------------
    def sample_X(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=(n, self.d))

    def sample(self, n: int, rng: np.random.Generator, delta: float = 0.0, noise: float = 0.0):
        X = self.sample_X(n, rng)
        y = self.label(X, delta)
        if noise > 0:
            flip = rng.random(n) < noise
            y = np.where(flip, 1 - y, y)
        return X, y.astype(np.int64)

    def delta_at(self, t: int, direction: int = +1) -> float:
        """Koncept w kroku t: drift liniowy, ograniczony przez Delta."""
        return float(direction * min(self.speed * t, self.default_bound))

    def steps_to_bound(self) -> int:
        return int(round(self.default_bound / self.speed))

    def disagreement(self, delta: float, n: int = 200_000, seed: int = 0) -> float:
        """Miara driftu niezalezna od parametryzacji: P[f(x;0) != f(x;delta)]."""
        X = self.sample_X(n, np.random.default_rng(seed))
        return float(np.mean(self.label(X, 0.0) != self.label(X, delta)))

    def class_balance(self, delta: float = 0.0, n: int = 100_000) -> float:
        X = self.sample_X(n, np.random.default_rng(1))
        return float(self.label(X, delta).mean())


class SEAStream(Stream):
    """SEA (Street & Kim 2001): 3 cechy w [0,10], tylko x1, x2 istotne, klasa 1 <=> x1 + x2 <= theta."""

    def __init__(self, theta0: float = 8.0, bound: float = 2.0, speed: float = 0.1):
        super().__init__(
            name="SEA", d=3, default_bound=bound, speed=speed,
            delta_label=r"drift progu $\Delta\theta$",
            feature_names=["x1", "x2", "x3"],
        )
        self.theta0 = theta0

    def label(self, X, delta=0.0):
        theta = self.theta0 + delta
        return (10.0 * (X[:, 0] + X[:, 1]) <= theta).astype(np.int64)


class HyperplaneStream(Stream):
    """Rotujaca hiperplaszczyzna (Hulten i in. 2001): klasa 1 <=> w.x >= 0.5 sum(w).

    Drift: n_drift cech ma wagi zmieniajace sie o delta * s_i, s_i in {-1,+1}.
    """

    def __init__(self, d: int = 10, n_drift: int = 3, bound: float = 0.3, speed: float = 0.015,
                 seed: int = 0):
        rng = np.random.default_rng(1000 + seed)
        super().__init__(
            name="Hyperplane", d=d, default_bound=bound, speed=speed,
            delta_label=r"drift wag $\Delta w$",
            feature_names=[f"x{i + 1}" for i in range(d)],
        )
        self.w0 = rng.uniform(0.45, 0.75, size=d)
        s = np.zeros(d)
        s[:n_drift] = np.where(np.arange(n_drift) % 2 == 0, 1.0, -1.0)   # +,-,+,...
        self.s = s

    def weights(self, delta: float) -> np.ndarray:
        return self.w0 + delta * self.s

    def label(self, X, delta=0.0):
        w = self.weights(delta)
        return (X @ w >= 0.5 * w.sum()).astype(np.int64)


def make_stream(name: str, seed: int = 0) -> Stream:
    name = name.lower()
    if name == "sea":
        return SEAStream()
    if name in ("hyperplane", "hp"):
        return HyperplaneStream(seed=seed)
    raise ValueError(name)


@dataclass
class DriftAssumption:
    """Zalozenie o *przyszlym* driftcie, uzywane przy trenowaniu (zakres i kierunek).

    delta in [lo, hi]:  symetryczne (-D, +D) = znamy tylko zakres, nie znamy kierunku;
                        jednostronne (0, +D) = znamy tez kierunek.
    """
    lo: float
    hi: float

    def grid(self, k: int = 11) -> np.ndarray:
        return np.linspace(self.lo, self.hi, k)

    @staticmethod
    def symmetric(bound: float) -> "DriftAssumption":
        return DriftAssumption(-bound, bound)

    @staticmethod
    def one_sided(bound: float, direction: int = +1) -> "DriftAssumption":
        return DriftAssumption(0.0, bound) if direction > 0 else DriftAssumption(-bound, 0.0)
