"""CZESC PODSTAWOWA: zapoznanie z tematem.

1. Jak wygladaja SEA i Hyperplane oraz ich ograniczony drift (fig_01).
2. Standardowy klasyfikator (MLP) zamrozony w t=0 - jak spada jego dokladnosc na dryfujacym strumieniu (fig_02).
3. DiCE (dice-ml) na standardowym klasyfikatorze: kontrfaktyk poprawny w t=0 przestaje byc poprawny
   po niewielkim driftcie (fig_03, fig_04).

Uruchomienie:  python experiments/run_basic.py [--quick]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TQDM_DISABLE", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cfd.cf import DiceGenerator
from cfd.evaluate import MethodSpec, run_grid, summarize
from cfd.metrics import delta_grid
from cfd.models import Classifier, TrainConfig
from cfd.plots import plot_stream_drift, savefig
from cfd.streams import make_stream

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "basic")
os.makedirs(OUT, exist_ok=True)


def fig_drift():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, name in zip(axes, ["sea", "hyperplane"]):
        plot_stream_drift(row, make_stream(name))
    savefig(fig, f"{OUT}/fig_01_drift_streams.png")


def fig_accuracy_over_time(seed=0):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = []
    for ax, name in zip(axes, ["sea", "hyperplane"]):
        st = make_stream(name, seed)
        rng = np.random.default_rng(seed)
        X, y = st.sample(5000, rng, 0.0, noise=0.10 if name == "sea" else 0.05)
        clf = Classifier("standard", st.d, TrainConfig(), seed=seed).fit(X, y, st)
        T = 2 * st.steps_to_bound()
        for direction, ls in [(+1, "-"), (-1, "--")]:
            acc = []
            for t in range(T + 1):
                Xt, yt = st.sample(3000, rng, st.delta_at(t, direction))
                acc.append((clf.predict(Xt) == yt).mean())
            ax.plot(range(T + 1), acc, ls, lw=2, label=f"drift {'+' if direction > 0 else '-'}")
            rows += [dict(stream=st.name, direction=direction, t=t, acc=a) for t, a in enumerate(acc)]
        ax.axvline(st.steps_to_bound(), color="k", ls=":", label=r"granica $\Delta$ osiągnięta")
        ax.set_title(f"{st.name}: dokładność zamrożonego klasyfikatora")
        ax.set_xlabel("krok strumienia t")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend()
    savefig(fig, f"{OUT}/fig_02_accuracy_over_time.png")
    pd.DataFrame(rows).to_csv(f"{OUT}/accuracy_over_time.csv", index=False)


def fig_sea_example(seed=0):
    """Pojedynczy przyklad: kontrfaktyk DiCE tuz za granica t=0 wypada za granice po driftcie."""
    st = make_stream("sea", seed)
    rng = np.random.default_rng(seed)
    X, y = st.sample(5000, rng, 0.0, noise=0.10)
    clf = Classifier("standard", st.d, TrainConfig(), seed=seed).fit(X, y, st)
    gen = DiceGenerator(clf, st, X, y, tau=0.5, seed=seed)
    Xt, yt = st.sample(5000, rng, 0.0)
    cand = np.where((yt == 0) & (clf.predict(Xt) == 0) & (10 * (Xt[:, 0] + Xt[:, 1]) < 11))[0][:6]
    Xq = Xt[cand]
    Xcf, ok = gen.generate(Xq, 1)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    gx = np.linspace(0, 1, 100)
    for d, c, ls in [(0.0, "k", "-"), (-st.default_bound, "tab:red", "--"), (st.default_bound, "tab:blue", "--")]:
        ax.plot(gx, d / 10 + 0 * gx - gx + (st.theta0 / 10), c=c, ls=ls,
                label=fr"granica $\delta={d:+.1f}$" + (" (t=0)" if d == 0 else ""))
    for q, c, o in zip(Xq, Xcf, ok):
        if o:
            ax.annotate("", xy=(c[0], c[1]), xytext=(q[0], q[1]), arrowprops=dict(arrowstyle="->", color="gray"))
            ax.scatter(*q[:2], c="tab:orange", s=40)
            ax.scatter(*c[:2], c="tab:green", s=40, marker="*")
    ax.scatter([], [], c="tab:orange", label="zapytanie"), ax.scatter([], [], c="tab:green", marker="*", label="kontrfaktyk DiCE")
    ax.set_xlim(0.1, 0.8), ax.set_ylim(0.1, 0.8)
    ax.set_xlabel("x1"), ax.set_ylabel("x2")
    ax.set_title("SEA: kontrfaktyki DiCE vs przesuwająca się granica")
    ax.legend(fontsize=8)
    savefig(fig, f"{OUT}/fig_03_sea_example.png")


def main(quick: bool):
    fig_drift()
    fig_accuracy_over_time()
    fig_sea_example()

    n_query, seeds = (15, [0]) if quick else (40, [0, 1, 2])
    spec = MethodSpec("Standard + DiCE", "standard", tau=0.5)
    cf_df, meta = run_grid(["sea", "hyperplane"], seeds, [spec], n_query=n_query)
    cf_df.to_csv(f"{OUT}/basic_cf.csv", index=False)
    table = summarize(cf_df, meta)
    table.to_csv(f"{OUT}/basic_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())

    # fig_04: przezywalnosc kontrfaktykow DiCE (najgorszy kierunek driftu), osobno dla kazdej klasy docelowej
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ["SEA", "Hyperplane"]):
        v = cf_df[(cf_df.stream == name) & cf_df.valid_model]
        grid = np.linspace(0, 2, 81)
        for sel, lab, ls in [(v, "wszystkie", "-"), (v[v.target == 1], "cel: klasa 1", "--"), (v[v.target == 0], "cel: klasa 0", ":")]:
            r = sel.Rrel.to_numpy()
            ax.plot(grid, [(r >= g - 1e-9).mean() for g in grid], ls, lw=2, label=lab)
        ax.axvline(1, color="k", ls=":", lw=0.8)
        ax.set_title(f"{name}: kontrfaktyki DiCE (model standardowy) po driftcie")
        ax.set_xlabel(r"zakres przyszłego driftu $r/\Delta$ (najgorszy kierunek)")
        ax.set_ylabel("odsetek wciąż poprawnych kontrfaktyków")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
    savefig(fig, f"{OUT}/fig_04_standard_cf_validity_decay.png")
    print(f"\nWyniki zapisane w {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
