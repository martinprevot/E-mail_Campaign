"""Report figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import MENS, WOMENS

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
COLORS = {MENS: "#2563eb", WOMENS: "#db2777"}
ARM_LABEL = {MENS: "E-mail Homme", WOMENS: "E-mail Femme"}
LEARNER_COLORS = {"S-learner": "#2563eb", "T-learner": "#16a34a", "X-learner": "#ea580c",
                  "Class transformation": "#7c3aed"}


def _style(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def effects_forest(effects: pd.DataFrame, path: Path):
    """Relative lift with 95 % CI, per metric and arm."""
    metrics = ["visit", "conversion", "spend"]
    labels = {"visit": "Visite", "conversion": "Achat", "spend": "Dépense ($)"}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = 0
    yticks, ylabels = [], []
    for m in metrics:
        for arm in [MENS, WOMENS]:
            r = effects[(effects.metric == m) & (effects.arm == arm)].iloc[0]
            ax.errorbar(r.rel_lift * 100, y, xerr=[[100 * (r.rel_lift - r.rel_ci_low)], [100 * (r.rel_ci_high - r.rel_lift)]],
                        fmt="o", color=COLORS[arm], capsize=3, ms=6)
            ax.text(r.rel_ci_high * 100 + 3, y, f"+{r.rel_lift:.0%}", va="center", fontsize=8, color=COLORS[arm])
            yticks.append(y)
            ylabels.append(f"{labels[m]} · {ARM_LABEL[arm]}")
            y -= 1
        y -= 0.6
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(yticks, ylabels)
    ax.set_xlabel("effet relatif vs groupe sans e-mail (%), IC 95 %")
    ax.set_title("Effet moyen de chaque e-mail (2 semaines après envoi)", loc="left", fontsize=11, color=INK)
    _style(ax, grid="x")
    _save(fig, path)


def effect_by_segment(seg: pd.DataFrame, path: Path):
    """Absolute uplift in visit rate by purchase-history segment."""
    segments = list(seg["segment"].unique())
    x = np.arange(len(segments))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, arm in enumerate([MENS, WOMENS]):
        s = seg[seg.arm == arm].set_index("segment").loc[segments]
        ax.bar(x + (i - 0.5) * 0.36, s["ate"] * 100, 0.36, yerr=1.96 * s["se"] * 100, capsize=3,
               color=COLORS[arm], label=ARM_LABEL[arm], error_kw=dict(ecolor=INK, lw=0.8))
    ax.set_xticks(x, segments)
    ax.set_ylabel("uplift taux de visite (points)")
    ax.set_title("L'e-mail Femme ne fait presque rien sur les clients qui n'achètent que de l'homme",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    _save(fig, path)


def qini_curves(curves: dict, path: Path):
    """curves[arm][learner] = (x, q); curves[arm]['_random'] = (x, q)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, arm in zip(axes, [MENS, WOMENS]):
        for name, (x, q) in curves[arm].items():
            if name.startswith("_"):
                continue
            ax.plot(x / x[-1] * 100, q, color=LEARNER_COLORS[name], linewidth=1.5, label=name)
        x, q = curves[arm]["_random"]
        ax.plot(x / x[-1] * 100, q, color=MUTED, linestyle="--", linewidth=1.2, label="ciblage aléatoire")
        ax.set_title(ARM_LABEL[arm], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("% de clients ciblés (triés par uplift prédit)")
        ax.set_ylabel("visites incrémentales (courbe de Qini)")
        _style(ax)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Courbes de Qini, prédictions out-of-fold (5 folds)", x=0.01, ha="left", fontsize=12, color=INK)
    _save(fig, path)


def uplift_bins(bins: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, arm in zip(axes, [MENS, WOMENS]):
        b = bins[arm]
        ax.bar(b.index, b["observed"] * 100, yerr=1.96 * b["se"] * 100, color=COLORS[arm], alpha=0.85,
               capsize=3, label="uplift observé (IC 95 %)", error_kw=dict(ecolor=INK, lw=0.8))
        ax.plot(b.index, b["predicted"] * 100, "o-", color=INK, ms=4, linewidth=1, label="uplift prédit")
        ax.set_xticks(b.index, [f"Q{i}" for i in b.index])
        ax.set_xlabel("quintile d'uplift prédit (Q1 = plus fort)")
        ax.set_title(ARM_LABEL[arm], loc="left", fontsize=11, color=INK)
        _style(ax)
    axes[0].set_ylabel("uplift taux de visite (points)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Calibration de l'uplift (S-learner) : le modèle sépare bien les clients pour l'e-mail Femme",
                 x=0.01, ha="left", fontsize=12, color=INK)
    _save(fig, path)


def targeting(curve_visit: pd.DataFrame, curve_spend: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, c, unit in [(axes[0], curve_visit, "visites incrémentales"), (axes[1], curve_spend, "CA incrémental ($)")]:
        ax.plot(c["share"] * 100, c["uplift_policy"], "o-", ms=3, color="#2563eb",
                label="ciblage par uplift (meilleur e-mail par client)")
        ax.plot(c["share"] * 100, c["random_blanket"], "--", color=MUTED,
                label="e-mail Homme à un échantillon aléatoire")
        ax.set_xlabel("% de clients contactés")
        ax.set_ylabel(unit)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Gain estimé (IPW, out-of-fold) selon le volume d'envoi", x=0.01, ha="left", fontsize=12, color=INK)
    _save(fig, path)


def bayes_posteriors(samples: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for arm, s in samples.items():
        ax.hist(s * 100, bins=80, density=True, alpha=0.55, color=COLORS[arm], label=ARM_LABEL[arm])
    ax.set_xlabel("hausse relative du taux de visite (%)")
    ax.set_yticks([])
    ax.set_title("Distribution a posteriori (Beta-Binomial) : P(Homme > Femme) ≈ 1", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, grid="x")
    _save(fig, path)


def power_curves(power: pd.DataFrame, n_actual: int, path: Path):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for col, label, color in [("visit", "Visite (base 10,6 %)", "#2563eb"), ("conversion", "Achat (base 0,57 %)", "#ea580c")]:
        ax.plot(power["n_per_arm"], power[col] * 100, color=color, linewidth=1.8, label=label)
    ax.axvline(n_actual, color=MUTED, linestyle=":", linewidth=1)
    ax.text(n_actual * 1.08, 0.08, f"taille réelle : {n_actual:,} / bras".replace(",", " "),
            fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("clients par bras (échelle log)")
    ax.set_ylabel("effet relatif minimum détectable (%)")
    ax.set_title("Analyse de puissance (α = 5 %, puissance 80 %)", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    _save(fig, path)
