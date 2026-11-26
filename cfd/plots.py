"""Wykresy do raportu."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .streams import Stream

COLORS = {}


def _color(name: str):
    if name not in COLORS:
        COLORS[name] = plt.cm.tab10(len(COLORS) % 10)
    return COLORS[name]


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_survival(ax, cf_df, stream_name: str, methods: list, title: str | None = None):
    """Krzywe przezycia: odsetek poprawnych kontrfaktykow, ktore dalej maja klase docelowa dla driftu <= r*Delta."""
    grid = np.linspace(0, 2, 81)
    sub = cf_df[(cf_df.stream == stream_name) & cf_df.valid_model]
    for m in methods:
        r = sub[sub.method == m].Rrel.to_numpy()
        if len(r) == 0:
            continue
        ax.plot(grid, [(r >= g - 1e-9).mean() for g in grid], label=m, color=_color(m), lw=2)
    ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel(r"zakres przyszłego driftu $r/\Delta$ (najgorszy kierunek)")
    ax.set_ylabel("odsetek kontrfaktyków wciąż poprawnych")
    ax.set_title(title or f"{stream_name}: przeżywalność kontrfaktyków")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)


def plot_stream_drift(axes, stream: Stream, n_pts: int = 1500, seed: int = 0):
    """Intuicja: jak zmienia sie granica decyzyjna (SEA: x1-x2) i odsetek zmienionych etykiet."""
    rng = np.random.default_rng(seed)
    ax1, ax2 = axes
    X = stream.sample_X(n_pts, rng)
    y0 = stream.label(X, 0.0)
    ax1.scatter(X[:, 0], X[:, 1], c=y0, cmap="coolwarm", s=6, alpha=0.5)
    gx = np.linspace(0, 1, 200)
    gy = np.linspace(0, 1, 200)
    G = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    Xg = np.full((len(G), stream.d), 0.5)
    Xg[:, :2] = G
    for d, ls in [(-stream.default_bound, "--"), (0.0, "-"), (stream.default_bound, "--")]:
        ax1.contour(gx, gy, stream.label(Xg, d).reshape(200, 200), levels=[0.5], colors="k",
                    linestyles=ls, linewidths=1.5)
    ax1.set_title(f"{stream.name}: granica dla $\\delta=-\\Delta, 0, +\\Delta$ (przekrój x1,x2)")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ds = np.linspace(-2 * stream.default_bound, 2 * stream.default_bound, 41)
    ax2.plot(ds / stream.default_bound, [stream.disagreement(d, n=50_000) for d in ds], lw=2)
    ax2.axvspan(-1, 1, color="tab:green", alpha=0.12, label=r"zakres ograniczony $|\delta|\leq\Delta$")
    ax2.set_xlabel(r"$\delta/\Delta$")
    ax2.set_ylabel(r"$P_x[f(x;0)\neq f(x;\delta)]$")
    ax2.set_title("Miara driftu: odsetek zmienionych etykiet")
    ax2.legend()
    ax2.grid(alpha=0.3)
