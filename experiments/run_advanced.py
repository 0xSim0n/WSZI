"""CZESC ZAAWANSOWANA: Counterfactuals by Design + Dual Decision Boundary vs standardowy klasyfikator (DiCE).

  A  porownanie glowne          (tabela + krzywe przezycia kontrfaktykow)
  B  kompromis dokladnosc / bliskosc / odpornosc (przeglad tau, alpha)
  C  bledne zalozenia o driftcie (za maly / za duzy zakres, zly kierunek)

Uruchomienie:  python experiments/run_advanced.py [--quick] [--only A B C]
Wyniki: results/advanced/*.csv, *.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TQDM_DISABLE", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cfd.evaluate import MethodSpec, run_grid, summarize
from cfd.plots import _color, plot_survival, savefig
from cfd.streams import make_stream

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "advanced")
os.makedirs(OUT, exist_ok=True)
STREAMS = ["sea", "hyperplane"]
SNAMES = {"sea": "SEA", "hyperplane": "Hyperplane"}
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

SHOW = ["stream", "method", "DiCE_success", "validity", "L1", "sparsity", "R_min", "robust_at_Delta",
        "robust_at_half", "acc_t0", "acc_drift"]


def save_table(table, name):
    table.to_csv(f"{OUT}/{name}.csv", index=False)
    with open(f"{OUT}/{name}.md", "w") as f:
        f.write(table.round(3).to_markdown(index=False) if hasattr(table, "to_markdown") else table.round(3).to_string())
    print(f"\n=== {name} ===")
    print(table[[c for c in SHOW if c in table.columns]].round(3).to_string(index=False))


# ----------------------------------------------------------------------------- A
def exp_A(n_query, seeds):
    specs = [
        MethodSpec("Standard (tau=0.5)", "standard", 0.5),
        MethodSpec("Standard (tau=0.8)", "standard", 0.8),
        MethodSpec("Standard, czyste etykiety (kontrola)", "standard", 0.5, clean=True),
        MethodSpec("CbD (tau=0.5)", "cbd", 0.5),
        MethodSpec("CbD (tau=0.9)", "cbd", 0.9),
        MethodSpec("Dual Boundary (tau=0.5)", "dual", 0.5),
    ]
    cf_df, meta = run_grid(STREAMS, seeds, specs, n_query=n_query)
    cf_df.to_csv(f"{OUT}/A_cf.csv", index=False)
    meta.to_csv(f"{OUT}/A_meta.csv", index=False)
    table = summarize(cf_df, meta)
    save_table(table, "A_main_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, s in zip(axes, STREAMS):
        plot_survival(ax, cf_df, SNAMES[s], [sp.name for sp in specs])
    axes[0].legend(fontsize=8, loc="upper right")
    savefig(fig, f"{OUT}/fig_A_survival.png")

    # histogram zakresu odpornosci
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, s in zip(axes, STREAMS):
        v = cf_df[(cf_df.stream == SNAMES[s]) & cf_df.valid_model]
        data = [v[v.method == sp.name].Rrel.to_numpy() for sp in specs]
        ax.boxplot(data, tick_labels=[sp.name.replace(" (", "\n(").replace(", ", ",\n") for sp in specs], showmeans=True)
        ax.axhline(1.0, color="k", ls=":")
        ax.set_ylabel(r"odporność $R/\Delta$")
        ax.set_title(f"{SNAMES[s]}: rozkład odporności czasowej kontrfaktyków")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3)
    savefig(fig, f"{OUT}/fig_A_box.png")
    return table


# ----------------------------------------------------------------------------- B
def exp_B(n_query, seeds):
    specs = []
    specs += [MethodSpec(f"Standard tau={t}", "standard", t) for t in (0.5, 0.6, 0.7, 0.8)]
    specs += [MethodSpec(f"CbD a=0.7 tau={t}", "cbd", t, alpha=0.7) for t in (0.5, 0.7, 0.8, 0.9)]
    specs += [MethodSpec(f"CbD a={a} tau=0.9", "cbd", 0.9, alpha=a) for a in (0.3, 0.5, 1.0)]
    specs += [MethodSpec(f"Dual tau={t}", "dual", t) for t in (0.5, 0.7, 0.9)]
    cf_df, meta = run_grid(STREAMS, seeds, specs, n_query=n_query)
    cf_df.to_csv(f"{OUT}/B_cf.csv", index=False)
    table = summarize(cf_df, meta)
    save_table(table, "B_tradeoff")
    fam = lambda m: "Standard" if m.startswith("Standard") else "CbD" if m.startswith("CbD") else "Dual Boundary"
    table["family"] = table.method.map(fam)
    marks = {"Standard": "o", "CbD": "s", "Dual Boundary": "*"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, s in enumerate(STREAMS):
        t = table[table.stream == SNAMES[s]]
        for c, (x, xl) in enumerate([("L1", "bliskość: średnie L1 (mniej = lepiej)"),
                                     ("acc_drift", "dokładność uśredniona po driftcie [-Δ,Δ]"),
                                     ("DiCE_success", "odsetek znalezionych kontrfaktyków")]):
            ax = axes[r, c]
            for f, g in t.groupby("family"):
                ax.scatter(g[x], g.R_min, marker=marks[f], s=110 if f == "Dual Boundary" else 60, label=f, color=_color(f))
                for _, row in g.iterrows():
                    ax.annotate(row.method.replace("CbD ", "").replace("Standard ", "").replace("Dual ", ""),
                                (row[x], row.R_min), fontsize=6, xytext=(3, 3), textcoords="offset points")
            ax.axhline(1.0, color="k", ls=":")
            ax.set_xlabel(xl)
            ax.set_ylabel(r"odporność czasowa $R/\Delta$")
            ax.set_title(SNAMES[s])
            ax.grid(alpha=0.3)
    axes[0, 0].legend()
    savefig(fig, f"{OUT}/fig_B_tradeoff.png")
    return table


# ----------------------------------------------------------------------------- C
def exp_C(n_query, seeds):
    specs = [MethodSpec("Standard", "standard", 0.5)]
    specs += [MethodSpec(f"Dual zakres={b}Δ", "dual", 0.5, bound_factor=b) for b in (0.5, 1.0, 2.0)]
    specs += [MethodSpec("Dual kierunek + (zły)", "dual", 0.5, assume="pos"),
              MethodSpec("Dual kierunek - (zły)", "dual", 0.5, assume="neg")]
    cf_df, meta = run_grid(STREAMS, seeds, specs, n_query=n_query)
    cf_df.to_csv(f"{OUT}/C_cf.csv", index=False)
    table = summarize(cf_df, meta)
    save_table(table, "C_misspecification")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for r, s in enumerate(STREAMS):
        st = make_stream(s)
        v = cf_df[(cf_df.stream == SNAMES[s]) & cf_df.valid_model]
        grid = np.linspace(0, 2, 81)
        ax = axes[r, 0]
        for m in ["Standard", "Dual zakres=0.5Δ", "Dual zakres=1.0Δ", "Dual zakres=2.0Δ"]:
            rr = v[v.method == m].Rrel.to_numpy()
            ax.plot(grid, [(rr >= g - 1e-9).mean() for g in grid], lw=2, label=m, color=_color(m))
        for b, m in [(0.5, "Dual zakres=0.5Δ"), (1.0, "Dual zakres=1.0Δ"), (2.0, "Dual zakres=2.0Δ")]:
            ax.axvline(b, color=_color(m), ls=":", lw=1)
        ax.set_title(f"{SNAMES[s]}: zalozony zakres driftu vs rzeczywisty (pionowe linie = założone Δ)")
        ax.set_xlabel(r"rzeczywisty zakres driftu $r/\Delta$")
        ax.set_ylabel("odsetek poprawnych kontrfaktyków")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[r, 1]
        for m, col, ls, c in [("Dual kierunek + (zły)", "R_plus", "-", "tab:blue"), ("Dual kierunek + (zły)", "R_minus", "--", "tab:blue"),
                              ("Dual zakres=1.0Δ", "R_plus", "-", "tab:green"), ("Dual zakres=1.0Δ", "R_minus", "--", "tab:green"),
                              ("Standard", "R_plus", "-", "tab:gray"), ("Standard", "R_minus", "--", "tab:gray")]:
            rr = v[v.method == m][col].to_numpy() / st.default_bound
            lab = f"{m} | drift {'+' if col == 'R_plus' else '-'}"
            ax.plot(grid, [(rr >= g - 1e-9).mean() for g in grid], ls, color=c, lw=2, label=lab)
        ax.set_title(f"{SNAMES[s]}: znajomość kierunku driftu (zał. drift $\\delta\\in[0,\\Delta]$)")
        ax.set_xlabel(r"rzeczywisty drift $r/\Delta$ w danym kierunku")
        ax.set_ylabel("odsetek poprawnych kontrfaktyków")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    savefig(fig, f"{OUT}/fig_C_misspecification.png")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=["A", "B", "C"])
    a = ap.parse_args()
    nq, seeds = (12, [0]) if a.quick else (40, [0, 1, 2])
    if "A" in a.only:
        exp_A(nq, seeds)
    if "B" in a.only:
        exp_B(max(nq // 2, 10) if not a.quick else 10, seeds[:2] if not a.quick else seeds)
    if "C" in a.only:
        exp_C(max(nq // 2, 10) if not a.quick else 10, seeds[:2] if not a.quick else seeds)
    print(f"\nWyniki zapisane w {OUT}")
