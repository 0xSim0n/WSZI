"""Protokol eksperymentalny: jeden przebieg = (strumien, ziarno, metoda) -> tabela kontrfaktykow z metrykami."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .cf import DiceGenerator
from .metrics import (accuracy_under_drift, delta_grid, proximity, robustness_radius, sparsity)
from .models import Classifier, TrainConfig
from .streams import DriftAssumption, make_stream

TRAIN_NOISE = {"sea": 0.10, "hyperplane": 0.05}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    kind: str                  # standard | cbd | dual
    tau: float = 0.5           # prog P(klasa docelowa) w DiCE (stopping_threshold)
    alpha: float = 0.7         # CbD: waga straty po drifcie
    beta: float = 1.0          # Dual: waga straty granicy odpornosci
    assume: str = "sym"        # zalozenie o driftcie: sym (zakres) | pos / neg (zakres + kierunek)
    bound_factor: float = 1.0  # zalozone Delta_train / prawdziwe Delta strumienia
    clean: bool = False        # kontrola: trenuj na etykietach bez szumu (oddziela efekt 'czystych etykiet' od driftu)
    dice_lr: float = 0.01      # DiCE: krok optymalizatora
    dice_pw: float = 0.5       # DiCE: proximity_weight (domyslnie 0.5)


def _assumption(spec: MethodSpec, stream) -> DriftAssumption | None:
    if spec.kind == "standard":
        return None
    b = spec.bound_factor * stream.default_bound
    return {"sym": DriftAssumption.symmetric(b),
            "pos": DriftAssumption.one_sided(b, +1),
            "neg": DriftAssumption.one_sided(b, -1)}[spec.assume]


def run_one(stream_name: str, seed: int, spec: MethodSpec, n_query: int = 40, n_train: int = 5000,
            max_iter: int = 1500, band_factor: float = 2.0):
    """Trenuje klasyfikator wg ``spec``, generuje kontrfaktyki DiCE i liczy metryki."""
    import torch
    torch.set_num_threads(1)
    os.environ["TQDM_DISABLE"] = "1"

    stream = make_stream(stream_name, seed)
    rng = np.random.default_rng(seed)
    X, y = stream.sample(n_train, rng, 0.0, noise=TRAIN_NOISE[stream_name.lower()])
    if spec.clean:
        y = stream.label(X, 0.0)
    clf = Classifier(spec.kind, stream.d, TrainConfig(alpha=spec.alpha, beta=spec.beta),
                     _assumption(spec, stream), seed=seed).fit(X, y, stream)

    # ---- jakosc predykcyjna (zamrozony model) ----
    rng_t = np.random.default_rng(10_000 + seed)
    Xt, yt = stream.sample(20_000, rng_t, 0.0)
    acc0, acc_drift = accuracy_under_drift(clf, stream, np.random.default_rng(20_000 + seed))

    # ---- zapytania: instancje ze strefy wrazliwej na drift (etykieta zmienia sie dla |delta| <= band*Delta) ----
    flips = np.zeros(len(Xt), dtype=bool)
    for d in np.linspace(-band_factor, band_factor, 21) * stream.default_bound:
        flips |= stream.label(Xt, d) != yt
    pred = clf.predict(Xt)

    gen = DiceGenerator(clf, stream, X, y, tau=spec.tau, max_iter=max_iter, learning_rate=spec.dice_lr,
                        proximity_weight=spec.dice_pw, seed=seed)
    deltas = delta_grid(stream)
    rows = []
    for target in (1, 0):
        src = 1 - target
        idx = np.where(flips & (yt == src) & (pred == src))[0][:n_query]
        Xq = Xt[idx]
        Xcf, ok = gen.generate(Xq, target)
        valid_model = ok & (clf.predict(Xcf) == target)
        valid_robust_head = ok & (clf.target_prob(Xcf, target) >= 0.5)
        l1, l2 = proximity(Xq, Xcf)
        sp = sparsity(Xq, Xcf)
        Rp, Rm, _, _ = robustness_radius(stream, Xcf, target, deltas)
        o0 = stream.label(Xcf, 0.0) == target
        for k in range(len(idx)):
            rows.append(dict(
                stream=stream.name, seed=seed, method=spec.name, target=target,
                dice_ok=bool(ok[k]), valid_model=bool(valid_model[k]),
                valid_robust_head=bool(valid_robust_head[k]),
                l1=l1[k], l2=l2[k], sparsity=sp[k],
                R_plus=Rp[k], R_minus=Rm[k], R_min=min(Rp[k], Rm[k]),
                Rrel=min(Rp[k], Rm[k]) / stream.default_bound,
                valid_oracle0=bool(o0[k]),
            ))
    df = pd.DataFrame(rows)
    meta = dict(stream=stream.name, seed=seed, method=spec.name, acc0=acc0, acc_drift=acc_drift)
    return df, meta


def _job(args):
    return run_one(*args[0], **args[1])


def run_grid(stream_names, seeds, specs, workers: int = 8, **kw):
    """Rownolegle uruchomienie calej siatki (strumien x ziarno x metoda). Zwraca (cf_df, meta_df)."""
    jobs = [((s, seed, spec), kw) for s in stream_names for seed in seeds for spec in specs]
    if workers <= 1:
        res = [_job(j) for j in jobs]
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_job, jobs))
    cf_df = pd.concat([r[0] for r in res], ignore_index=True)
    meta = pd.DataFrame([r[1] for r in res])
    return cf_df, meta


def summarize(cf_df: pd.DataFrame, meta: pd.DataFrame, by=("stream", "method")) -> pd.DataFrame:
    """Tabela zbiorcza. Metryki kontrfaktykow liczone po poprawnych (wg wlasnego modelu) kontrfaktykach."""
    by = list(by)
    g = cf_df.groupby(by, sort=False)
    succ = g.agg(DiCE_success=("dice_ok", "mean"), validity=("valid_model", "mean"))
    v = cf_df[cf_df.valid_model]
    gv = v.groupby(by, sort=False)
    qual = gv.agg(L1=("l1", "mean"), L2=("l2", "mean"), sparsity=("sparsity", "mean"),
                  R_min=("Rrel", "mean"), R_median=("Rrel", "median"),
                  robust_at_Delta=("Rrel", lambda r: float((r >= 1.0).mean())),
                  robust_at_half=("Rrel", lambda r: float((r >= 0.5).mean())))
    acc = meta.groupby(by, sort=False).agg(acc_t0=("acc0", "mean"), acc_drift=("acc_drift", "mean"))
    out = succ.join(qual).join(acc)
    return out.reset_index()


def survival_curve(cf_df: pd.DataFrame, stream, grid_rel: np.ndarray | None = None):
    """S(r) = odsetek poprawnych kontrfaktykow z R_min >= r (r w jednostkach Delta)."""
    grid_rel = np.linspace(0, 2, 81) if grid_rel is None else grid_rel
    v = cf_df[cf_df.valid_model]
    return grid_rel, np.array([(v.Rrel >= r - 1e-9).mean() for r in grid_rel])
