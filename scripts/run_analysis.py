"""End-to-end analysis: A/B test reading -> uplift models -> targeting policy.

Usage:  python scripts/run_analysis.py
Outputs go to reports/ (results.json, CSV tables, figures/).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campaign_uplift import experiment as ex  # noqa: E402
from campaign_uplift import plots  # noqa: E402
from campaign_uplift.crossfit import cross_fit_all  # noqa: E402
from campaign_uplift.data import ARMS, CONTROL, MENS, OUTCOMES, WOMENS, feature_matrix, load  # noqa: E402
from campaign_uplift.learners import LEARNERS  # noqa: E402
from campaign_uplift.metrics import qini_auc_score, qini_curve, uplift_at_k, uplift_by_bin  # noqa: E402
from campaign_uplift.policy import best_arm_policy, ipw_diff_ci, random_policy, targeting_curve  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
EMAIL_ARMS = [MENS, WOMENS]


def part1_experiment(df, X, out):
    print("== Part 1: experiment analysis")
    counts = df["arm"].value_counts().reindex(ARMS)
    out["arm_sizes"] = counts.to_dict()
    out["srm"] = ex.srm_test(counts)
    smd = ex.standardized_mean_differences(X, df["arm"], CONTROL)
    out["max_abs_smd"] = float(smd.abs().max().max())
    print(f"  SRM p={out['srm']['p_value']:.3f}, max |SMD|={out['max_abs_smd']:.3f}")

    # Average effects, unadjusted and regression-adjusted
    X_lin = feature_matrix(df, drop_first=True)
    rows = []
    for m in OUTCOMES:
        yc = df.loc[df.arm == CONTROL, m]
        for arm in EMAIL_ARMS:
            yt = df.loc[df.arm == arm, m]
            e = ex.diff_in_means(yt, yc, metric=m, arm=arm).to_dict()
            pair = df.arm.isin([arm, CONTROL])
            adj = ex.regression_adjusted_ate(df.loc[pair, m], (df.loc[pair, "arm"] == arm).astype(int), X_lin[pair])
            e.update({f"adj_{k}": v for k, v in adj.items()})
            e["variance_reduction"] = 1 - (adj["se"] / e["se"]) ** 2
            if m == "spend":
                e["boot_ci_low"], e["boot_ci_high"] = ex.bootstrap_ci(yt, yc)
            rows.append(e)
    # Head-to-head: Mens vs Womens e-mail
    for m in OUTCOMES:
        e = ex.diff_in_means(df.loc[df.arm == MENS, m], df.loc[df.arm == WOMENS, m],
                             metric=m, arm="mens_vs_womens").to_dict()
        rows.append(e)
    effects = pd.DataFrame(rows)
    effects["p_holm"] = ex.holm(effects["p_value"])
    effects.to_csv(REPORTS / "effects.csv", index=False)
    out["effects"] = effects.round(6).to_dict(orient="records")
    for _, r in effects.iterrows():
        print(f"  {r.metric:<10} {r.arm:<15} ATE={r.ate:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] "
              f"lift={r.rel_lift:+.1%} p_holm={r.p_holm:.2g}")

    # Heterogeneity by purchase-history segment (pre-registered style breakdown)
    seg_rows = []
    for seg, g in df.groupby("buyer_type"):
        yc = g.loc[g.arm == CONTROL, "visit"]
        for arm in EMAIL_ARMS:
            e = ex.diff_in_means(g.loc[g.arm == arm, "visit"], yc, "visit", arm)
            seg_rows.append({"segment": seg, "arm": arm, "n": len(g), "ate": e.ate, "se": e.se,
                             "control_rate": e.control_mean})
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(REPORTS / "effects_by_segment.csv", index=False)
    out["effects_by_segment"] = seg.round(5).to_dict(orient="records")

    # Bayesian reading
    bayes, samples = {}, {}
    nc, sc = (df.arm == CONTROL).sum(), df.loc[df.arm == CONTROL, "visit"].sum()
    for arm in EMAIL_ARMS:
        nt, st = (df.arm == arm).sum(), df.loc[df.arm == arm, "visit"].sum()
        b = ex.bayesian_binomial(st, nt, sc, nc)
        samples[arm] = b.pop("samples_lift")
        bayes[arm] = b
    nm, sm_ = (df.arm == MENS).sum(), df.loc[df.arm == MENS, "visit"].sum()
    nw, sw = (df.arm == WOMENS).sum(), df.loc[df.arm == WOMENS, "visit"].sum()
    b = ex.bayesian_binomial(sm_, nm, sw, nw)
    b.pop("samples_lift")
    bayes["mens_vs_womens"] = b
    out["bayesian_visit"] = bayes

    # Power analysis
    n_arm = int(counts[CONTROL])
    p_visit, p_conv = df.loc[df.arm == CONTROL, ["visit", "conversion"]].mean()
    sd_spend = df.loc[df.arm == CONTROL, "spend"].std()
    out["power"] = {
        "n_per_arm": n_arm,
        "mde_visit_abs": ex.mde_proportion(p_visit, n_arm),
        "mde_visit_rel": ex.mde_proportion(p_visit, n_arm) / p_visit,
        "mde_conversion_abs": ex.mde_proportion(p_conv, n_arm),
        "mde_conversion_rel": ex.mde_proportion(p_conv, n_arm) / p_conv,
        "mde_spend_abs": ex.mde_mean(sd_spend, n_arm),
        "n_needed_conversion_lift_20pct": ex.n_per_arm_proportion(p_conv, 0.20),
        "n_needed_visit_lift_5pct": ex.n_per_arm_proportion(p_visit, 0.05),
    }
    grid = np.unique(np.logspace(3, 6, 40).astype(int))
    power = pd.DataFrame({
        "n_per_arm": grid,
        "visit": [ex.mde_proportion(p_visit, n) / p_visit for n in grid],
        "conversion": [ex.mde_proportion(p_conv, n) / p_conv for n in grid],
    })

    plots.effects_forest(effects[effects.arm.isin(EMAIL_ARMS)], FIG / "01_average_effects.png")
    plots.effect_by_segment(seg, FIG / "02_effect_by_segment.png")
    plots.bayes_posteriors(samples, FIG / "06_bayesian_posteriors.png")
    plots.power_curves(power, n_arm, FIG / "07_power_analysis.png")


def part2_uplift(df, X, out):
    print("== Part 2: uplift models (target = visit, 5-fold cross-fitting)")
    taus, rows, curves = {}, [], {a: {} for a in EMAIL_ARMS}
    for learner in LEARNERS:
        tau = cross_fit_all(X, df["arm"], df["visit"], learner, EMAIL_ARMS)
        taus[learner.name] = tau
        for arm in EMAIL_ARMS:
            m = df.arm.isin([arm, CONTROL]).to_numpy()
            y, t, s = df.loc[m, "visit"].to_numpy(), (df.loc[m, "arm"] == arm).astype(int).to_numpy(), tau.loc[m, arm].to_numpy()
            rows.append({"learner": learner.name, "arm": arm, "qini_auc": qini_auc_score(y, t, s),
                         "uplift_at_10": uplift_at_k(y, t, s, 0.1), "uplift_at_20": uplift_at_k(y, t, s, 0.2),
                         "uplift_at_30": uplift_at_k(y, t, s, 0.3),
                         "uplift_overall": y[t == 1].mean() - y[t == 0].mean()})
            curves[arm][learner.name] = qini_curve(y, t, s)
            if "_random" not in curves[arm]:
                x, q = curves[arm][learner.name]
                curves[arm]["_random"] = (np.array([0, x[-1]]), np.array([0, q[-1]]))
    table = pd.DataFrame(rows)
    table.to_csv(REPORTS / "uplift_models.csv", index=False)
    out["uplift_models"] = table.round(5).to_dict(orient="records")
    print(table.round(4).to_string(index=False))

    best = "S-learner"
    bins = {}
    for arm in EMAIL_ARMS:
        m = df.arm.isin([arm, CONTROL]).to_numpy()
        bins[arm] = uplift_by_bin(df.loc[m, "visit"].to_numpy(), (df.loc[m, "arm"] == arm).astype(int).to_numpy(),
                                  taus[best].loc[m, arm].to_numpy(), bins=5)
    out["uplift_by_quintile"] = {a: b.round(5).reset_index().to_dict(orient="records") for a, b in bins.items()}

    plots.qini_curves(curves, FIG / "03_qini_curves.png")
    plots.uplift_bins(bins, FIG / "04_uplift_by_quintile.png")
    return taus[best]


def part3_policy(df, tau, out):
    print("== Part 3: targeting policy (IPW off-policy evaluation)")
    prop = df["arm"].value_counts(normalize=True).reindex(ARMS).to_dict()
    n = len(df)
    none = np.full(n, CONTROL)
    policies = {
        "no_email": none,
        "all_womens": np.full(n, WOMENS),
        "all_mens": np.full(n, MENS),
        "rule_womens_to_womens_only_buyers": np.where(df["buyer_type"] == "Femme uniquement", WOMENS, MENS),
        "uplift_best_email_all": best_arm_policy(tau, CONTROL, share=1.0),
        "random_mens_30pct": random_policy(n, MENS, 0.3, CONTROL, ARMS),
        "uplift_top_30pct": best_arm_policy(tau, CONTROL, share=0.3),
    }
    res = {}
    for name, pol in policies.items():
        share = 0.3 if name == "random_mens_30pct" else float((np.asarray(pol) != CONTROL).mean())
        r = {"share_emailed": share}
        for m in ["visit", "spend"]:
            est, lo, hi = ipw_diff_ci(df[m], df["arm"], pol, none, prop)
            r[f"incr_{m}_total"] = est * n
            r[f"incr_{m}_total_ci"] = [lo * n, hi * n]
        res[name] = r
        print(f"  {name:<34} emailed={share:4.0%}  +visits={r['incr_visit_total']:6.0f} "
              f"[{r['incr_visit_total_ci'][0]:6.0f}, {r['incr_visit_total_ci'][1]:6.0f}]  "
              f"+revenue=${r['incr_spend_total']:7.0f} [{r['incr_spend_total_ci'][0]:7.0f}, {r['incr_spend_total_ci'][1]:7.0f}]")
    out["policies"] = res

    comparisons = {
        "uplift_top_30pct_vs_random_mens_30pct": ("uplift_top_30pct", "random_mens_30pct"),
        "uplift_best_email_all_vs_all_mens": ("uplift_best_email_all", "all_mens"),
        "rule_vs_all_mens": ("rule_womens_to_womens_only_buyers", "all_mens"),
    }
    comp = {}
    for name, (a, b) in comparisons.items():
        comp[name] = {}
        for m in ["visit", "spend"]:
            est, lo, hi = ipw_diff_ci(df[m], df["arm"], policies[a], policies[b], prop)
            comp[name][m] = {"diff_total": est * n, "ci": [lo * n, hi * n]}
        print(f"  {name:<40} visits {comp[name]['visit']['diff_total']:+6.0f} "
              f"[{comp[name]['visit']['ci'][0]:+.0f}, {comp[name]['visit']['ci'][1]:+.0f}]  "
              f"revenue {comp[name]['spend']['diff_total']:+7.0f} "
              f"[{comp[name]['spend']['ci'][0]:+.0f}, {comp[name]['spend']['ci'][1]:+.0f}]")
    out["policy_comparisons"] = comp

    cv = targeting_curve(df["visit"], df["arm"], tau, CONTROL, prop, MENS)
    cs = targeting_curve(df["spend"], df["arm"], tau, CONTROL, prop, MENS)
    cv.merge(cs, on="share", suffixes=("_visit", "_spend")).to_csv(REPORTS / "targeting_curve.csv", index=False)
    full = cv["random_blanket"].iloc[-1]
    out["targeting_visit_capture"] = {
        f"{int(round(s * 100))}pct": float(cv.loc[np.isclose(cv.share, s), "uplift_policy"].iloc[0] / full)
        for s in [0.1, 0.2, 0.3, 0.5]
    }
    print("  share of 'Mens to all' incremental visits captured:", {k: round(v, 3) for k, v in out["targeting_visit_capture"].items()})
    plots.targeting(cv, cs, FIG / "05_targeting_curve.png")


def main():
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    X = feature_matrix(df)
    out = {"n_customers": len(df)}
    part1_experiment(df, X, out)
    tau = part2_uplift(df, X, out)
    part3_policy(df, tau, out)
    (REPORTS / "results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
