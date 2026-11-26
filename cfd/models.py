"""Klasyfikatory: standardowy, Counterfactuals-by-Design (CbD) i Dual Decision Boundary.

Pomysl (CbD): wlasnosc, ktorej oczekujemy od kontrfaktykow - przetrwanie ograniczonego driftu -
wbudowujemy w funkcje straty klasyfikatora, a nie w generator kontrfaktykow.

Niech ``Delta_k`` (k = 1..K) beda *przewidywanymi* wartosciami driftu z zalozonego zbioru
(``DriftAssumption``), a ``y_k(x) = f(x; phi(Delta_k))`` etykieta x w koncepcie Delta_k.
Z nich budujemy etykiety "po drifcie":

    q_mean(x) = mean_k y_k(x)        - ulamek scenariuszy, w ktorych x jest klasy 1
    q_all1(x) = min_k  y_k(x)        - x jest klasy 1 we WSZYSTKICH scenariuszach (region bezpieczny dla klasy 1)
    q_all0(x) = 1 - max_k y_k(x)     - x jest klasy 0 we WSZYSTKICH scenariuszach (region bezpieczny dla klasy 0)

1. ``standard`` : L = BCE(f(x), y)
2. ``cbd``      : L = (1-alpha) * BCE(f(x), y) + alpha * BCE(f(x), q_mean(x))
                  p(x) jest "rozmytym" prawdopodobienstwem po driftcie: p >= tau oznacza, ze
                  >= tau scenariuszy driftu zachowuje klase -> prog tau w DiCE (stopping_threshold)
                  wybiera region odporny. alpha steruje kompromisem dokladnosc <-> odpornosc.
3. ``dual``     : dwie powiazane granice - glowa klasyfikacyjna f_c (L = BCE(f_c, y)) oraz glowy
                  odpornosci f_r1, f_r0 (L = beta * [BCE(f_r1, q_all1) + BCE(f_r0, q_all0)]).
                  Predykcja = f_c; kontrfaktyk ma dojsc do f_r{target} >= tau (granica odpornosci),
                  a nie tylko przekroczyc f_c = 0.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .streams import DriftAssumption, Stream


@dataclass
class TrainConfig:
    hidden: tuple = (64, 64)
    epochs: int = 60
    lr: float = 3e-3
    batch: int = 256
    weight_decay: float = 1e-5
    alpha: float = 0.7         # CbD: waga straty "po drifcie"
    beta: float = 1.0          # Dual: waga straty granicy odpornosci
    n_scenarios: int = 11      # K - liczba przewidywanych scenariuszy driftu


def _mlp(d_in: int, d_out: int, hidden) -> nn.Sequential:
    layers, prev = [], d_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU()]   # gladka aktywacja: ReLU daje plaskie obszary (gradient 0) -> DiCE utyka
        prev = h
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


class _TargetProb(nn.Module):
    """Adapter dla DiCE: zwraca P(klasa docelowa | x) jako tensor (n, 1).

    DiCE (wersja gradientowa) z ``desired_class=1`` i ``stopping_threshold=tau`` szuka x', dla ktorego
    zwrocona wartosc >= tau. Podajac P(target) zamiast P(klasa 1) ujednolicamy oba kierunki 0->1 i 1->0.
    """

    def __init__(self, clf: "Classifier", target: int):
        super().__init__()
        self.clf, self.target = clf, target

    def forward(self, x):
        z = self.clf.target_logit_t(x, self.target)
        # Liniowy "clamp z wyciekiem": identycznosc dla |z| <= 6 (p in [0.0025, 0.9975]), poza tym nachylenie
        # 0.05. Zapobiega nasyceniu sigmoidy w float32 (p == 0/1 -> log-odds = +-inf -> NaN w DiCE),
        # stosowane jednakowo do wszystkich modeli; nie zmienia progow tau <= 0.99.
        zc = z.clamp(-6.0, 6.0)
        z = zc + 0.05 * (z - zc)
        return torch.sigmoid(z).unsqueeze(1)


class Classifier:
    KINDS = ("standard", "cbd", "dual")

    def __init__(self, kind: str, d: int, cfg: TrainConfig | None = None,
                 assumption: DriftAssumption | None = None, seed: int = 0):
        assert kind in self.KINDS
        self.kind, self.d, self.cfg, self.assumption, self.seed = kind, d, cfg or TrainConfig(), assumption, seed
        if kind != "standard" and assumption is None:
            raise ValueError("cbd/dual wymagaja DriftAssumption")
        torch.manual_seed(seed)
        self.net = _mlp(d, 3 if kind == "dual" else 1, self.cfg.hidden)

    # ------------------------------------------------------------------ trening
    def fit(self, X: np.ndarray, y: np.ndarray, stream: Stream | None = None) -> "Classifier":
        cfg = self.cfg
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        extra = None
        if self.kind != "standard":
            grid = self.assumption.grid(cfg.n_scenarios)
            Y = np.stack([stream.label(X, dd) for dd in grid], axis=1)           # (N, K)
            if self.kind == "cbd":
                extra = torch.tensor(Y.mean(1), dtype=torch.float32).unsqueeze(1)
            else:
                extra = torch.tensor(np.stack([Y.min(1), 1 - Y.max(1)], 1), dtype=torch.float32)

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        g = torch.Generator().manual_seed(self.seed)
        n = len(Xt)
        self.net.train()
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, cfg.batch):
                idx = perm[i:i + cfg.batch]
                out = self.net(Xt[idx])
                loss = self._loss(out, yt[idx], None if extra is None else extra[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        self.net.eval()
        return self

    def _loss(self, out, y, extra):
        cfg = self.cfg
        if self.kind == "standard":
            return F.binary_cross_entropy_with_logits(out[:, 0], y)
        if self.kind == "cbd":
            l_cur = F.binary_cross_entropy_with_logits(out[:, 0], y)
            l_drift = F.binary_cross_entropy_with_logits(out[:, 0], extra[:, 0])
            return (1 - cfg.alpha) * l_cur + cfg.alpha * l_drift
        l_cls = F.binary_cross_entropy_with_logits(out[:, 0], y)
        l_rob = F.binary_cross_entropy_with_logits(out[:, 1:], extra)
        return l_cls + cfg.beta * l_rob

    # ---------------------------------------------------------------- predykcja
    @torch.no_grad()
    def _raw(self, X) -> torch.Tensor:
        return self.net(torch.as_tensor(X, dtype=torch.float32))

    def predict_proba(self, X) -> np.ndarray:
        """P(klasa 1) wg glowy klasyfikacyjnej (dla cbd = rozmyte prawdopodobienstwo po driftcie)."""
        return torch.sigmoid(self._raw(X)[:, 0]).numpy()

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int64)

    def target_logit_t(self, x: torch.Tensor, target: int) -> torch.Tensor:
        """Logit P(klasa docelowa | x) (tensor z gradientem)."""
        if x.dim() == 1:                       # DiCE potrafi podac pojedynczy wektor
            x = x.unsqueeze(0)
        out = self.net(x)
        if self.kind == "dual":                # glowa odpornosci dla klasy docelowej
            return out[:, 1] if target == 1 else out[:, 2]
        return out[:, 0] if target == 1 else -out[:, 0]

    def target_prob_t(self, x: torch.Tensor, target: int) -> torch.Tensor:
        return torch.sigmoid(self.target_logit_t(x, target))

    def target_prob(self, X, target: int) -> np.ndarray:
        with torch.no_grad():
            return self.target_prob_t(torch.as_tensor(X, dtype=torch.float32), target).numpy()

    def dice_module(self, target: int) -> nn.Module:
        m = _TargetProb(self, target)
        m.eval()
        return m
