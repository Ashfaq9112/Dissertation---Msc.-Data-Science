"""
gamma_bench.results.analysis
============================

Builds statistics and visualizations from RunRecords ONLY. Each function maps
to a pre-registered hypothesis (H1-H6) or operator tier (T1-T3), so the output
of the design is a fixed set of tables/plots the moment runs land.

Dependency-light: pandas for groupbys, matplotlib for the plot shells. Swap or
extend freely; the contract is the RunRecord schema, not these functions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def to_frame(records: list[dict]) -> pd.DataFrame:
    """Flatten RunRecords to a tidy frame. Nested payloads get prefixed cols."""
    rows = []
    for r in records:
        row = {k: r[k] for k in
               ("run_id", "stage", "model", "method", "family",
                "budget", "seed", "rotation", "geometry_terms")}
        row["ppl_wikitext2"] = r["perf"].get("ppl_wikitext2")
        row["ppl_c4"] = r["perf"].get("ppl_c4")
        row["effective_bits"] = r["perf"].get("effective_bits")
        row["cka_mean"] = r["geom"].get("cka_mean")
        row["knn_mean"] = r["geom"].get("knn_mean")
        for task, acc in r["perf"].get("zeroshot", {}).items():
            row[f"zs_{task}"] = acc
        op = r["operator"]
        row["tokens_per_gpu"] = op.get("tokens_per_gpu")
        row["arith_intensity"] = op.get("arith_intensity")
        row["bound_regime"] = op.get("bound_regime")
        row["h_minus_sigma"] = r["hessian_probe"].get("h_minus_sigma_norm")
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Statistics with seed variance (mean +/- std required for credibility)
# ----------------------------------------------------------------------
def aggregate_seeds(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """Collapse seeds -> mean/std. Groups by everything except seed."""
    keys = ["stage", "model", "method", "family", "budget",
            "rotation", "geometry_terms"]
    g = df.groupby(keys)[value_cols]
    out = g.agg(["mean", "std", "count"])
    out.columns = ["_".join(c) for c in out.columns]
    return out.reset_index()


# ----------------------------------------------------------------------
# H1/H2/H3 : crux ablation -- gamma vs the Hessian ladder L0..L4
# ----------------------------------------------------------------------
def crux_ladder_table(df: pd.DataFrame, model: str, budget: float) -> pd.DataFrame:
    """
    For a (model, budget): gamma_lgeo vs ladder_L0..L4 on geometry + ppl.
    Verdict logic (interpretation, not computed here):
      advantage shrinks toward L4  -> H1 (cheap surrogate)
      advantage persists at L4     -> H2 (genuine surplus)
      ranking flips on the ladder  -> H3 (approximation-sensitive)
    """
    sub = df[(df.model == model) & (df.budget == budget)]
    sub = sub[sub.method.str.startswith(("gamma", "ladder"))]
    agg = aggregate_seeds(sub, ["cka_mean", "knn_mean", "ppl_wikitext2"])
    order = {"ladder_L0": 0, "ladder_L1": 1, "ladder_L2": 2,
             "ladder_L3": 3, "ladder_L4": 4, "gamma_lgeo": 5}
    agg["__o"] = agg.method.map(order).fillna(99)
    return agg.sort_values("__o").drop(columns="__o")


# ----------------------------------------------------------------------
# H4/H5/H6 : rotation interaction
# ----------------------------------------------------------------------
def rotation_interaction_table(df: pd.DataFrame, model: str,
                               budget: float) -> pd.DataFrame:
    """
    rotation+uniform vs rotation+gamma at matched budget.
      gamma adds nothing on rotated weights      -> H4 (rotation absorbs)
      gamma wins, gain in high-C(x) layers       -> H5 (complementary)
      rotation better ppl, worse geometry        -> H6 (antagonistic)
    """
    sub = df[(df.model == model) & (df.budget == budget) & (df.rotation)]
    return aggregate_seeds(sub, ["cka_mean", "knn_mean", "ppl_wikitext2"])


# ----------------------------------------------------------------------
# T3 : geometry -> task-quality link (the high-value operator unlock)
# ----------------------------------------------------------------------
def geometry_quality_regression(df: pd.DataFrame, task_col: str) -> dict:
    """
    Across the bit-width sweep, does CKA / kNN predict downstream accuracy?
    Returns simple correlations + OLS slope. If predictive, T3 claim unlocks:
    'gamma extends usable compression range at fixed quality floor.'
    """
    d = df.dropna(subset=["cka_mean", "knn_mean", task_col])
    if len(d) < 3:
        return {"status": "insufficient_data", "n": len(d)}
    out = {"n": len(d)}
    for g in ("cka_mean", "knn_mean"):
        x, y = d[g].values, d[task_col].values
        r = float(np.corrcoef(x, y)[0, 1])
        slope = float(np.polyfit(x, y, 1)[0])
        out[g] = {"pearson_r": r, "ols_slope": slope}
    return out


# ======================================================================
# Visualization shells (matplotlib). Each returns a fig; callers save.
# ======================================================================
def _lazy_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_ladder_geometry(crux_df: pd.DataFrame, metric="cka_mean_mean",
                         err="cka_mean_std"):
    """H1-H3 figure: geometry vs ladder rung, gamma overlaid, with error bars."""
    plt = _lazy_plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(crux_df["method"], crux_df[metric],
                yerr=crux_df.get(err), marker="o", capsize=3)
    ax.set_xlabel("allocation method (ladder rung -> gamma)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Crux ablation: geometry preservation across Hessian ladder")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def plot_pareto(df: pd.DataFrame, model: str):
    """PPL vs geometry Pareto, one point per method; gamma highlighted."""
    plt = _lazy_plt()
    sub = df[df.model == model]
    fig, ax = plt.subplots(figsize=(6, 4))
    for fam, grp in sub.groupby("family"):
        ax.scatter(grp["ppl_wikitext2"], grp["cka_mean"], label=fam, s=40)
    ax.set_xlabel("PPL (WikiText-2, lower better)")
    ax.set_ylabel("CKA (higher better)")
    ax.set_title(f"Performance vs geometry Pareto -- {model}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_roofline(df: pd.DataFrame, peak_flops: float, peak_bw: float,
                  model: str):
    """T2 figure: place compressed models on the accelerator roofline.
    Explicitly a MODEL, not a measurement -- title says so."""
    plt = _lazy_plt()
    sub = df[df.model == model].dropna(subset=["arith_intensity"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ridge = peak_flops / peak_bw
    xs = np.logspace(-1, 3, 200)
    ax.plot(xs, np.minimum(peak_flops, peak_bw * xs), "k--", lw=1,
            label="roofline")
    ax.axvline(ridge, color="grey", ls=":", lw=1, label="ridge point")
    for _, r in sub.iterrows():
        ax.scatter(r["arith_intensity"],
                   min(peak_flops, peak_bw * r["arith_intensity"]),
                   s=40, label=r["method"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("attainable FLOP/s")
    ax.set_title(f"Roofline (MODELED, kernel-dependent) -- {model}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_geometry_quality(df: pd.DataFrame, task_col: str):
    """T3 figure: scatter of task accuracy vs CKA across bit-widths + fit."""
    plt = _lazy_plt()
    d = df.dropna(subset=["cka_mean", task_col])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d["cka_mean"], d[task_col], s=40)
    if len(d) >= 2:
        m, b = np.polyfit(d["cka_mean"], d[task_col], 1)
        xs = np.linspace(d["cka_mean"].min(), d["cka_mean"].max(), 50)
        ax.plot(xs, m * xs + b, "r-", lw=1)
    ax.set_xlabel("CKA (geometry preserved)")
    ax.set_ylabel(f"task accuracy ({task_col})")
    ax.set_title("T3: does geometry predict task quality?")
    fig.tight_layout()
    return fig
