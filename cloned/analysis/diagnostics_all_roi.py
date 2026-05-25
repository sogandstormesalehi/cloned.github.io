from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

OUT_DIR  = Path("./cloned/results")
DIAGDIR  = OUT_DIR / "diagnostics"
DIAGDIR.mkdir(parents=True, exist_ok=True)

STATELESS = "genetic_clip_stateless"
EXPLORER  = "genetic_clip_explorer"
ADAPTIVE  = "genetic_clip_adaptive_salience"
ALGOS     = [STATELESS, EXPLORER, ADAPTIVE]
ALGO_LABEL = {
    STATELESS: "Stateless",
    EXPLORER:  "Classic state-aware",
    ADAPTIVE:  "Adaptive state-aware",
}
ALGO_COLOR = {STATELESS: "#d6604d", EXPLORER: "#4dac26", ADAPTIVE: "#2166ac"}

ROI_CAL_KEY = {
    "V1v": "V1v", "PPA": "PPA", "FFA-1": "FFA-1", "EBA": "EBA",
    "OFA": "OFA", "OPA": "OPA", "FBA-1": "FBA-1", "RSC": "RSC",
    "aTL-faces": "aTL-faces",
}

TWO_ROI_CONFIGS = [
    {"max_evals": 150, "midpoint": 0.60},
    {"max_evals": 200, "midpoint": 0.60},
]
TWO_ROI_ALL_PAIRS = [
    ("FFA-1", "PPA"), ("OFA",  "FFA-1"), ("EBA",   "OFA"),
    ("V1v",   "PPA"), ("OPA",  "PPA"),   ("FBA-1", "EBA"),
    ("EBA",   "PPA"), ("V1v",  "FFA-1"), ("PPA",   "V1v"),
    ("PPA",   "OFA"),
]

THREE_ROI_CONFIGS = [
    {"max_evals": 250, "drift_1": 0.40, "drift_2": 0.70},
    {"max_evals": 300, "drift_1": 0.40, "drift_2": 0.70},
]
THREE_ROI_ALL_TRIPLETS = [
    ("V1v",   "OFA",   "FFA-1"),
    ("FFA-1", "PPA",   "V1v"),
    ("OFA",   "FFA-1", "PPA"),
    ("OFA",   "FFA-1", "aTL-faces"),
    ("OPA",   "PPA",   "RSC"),
]

FOUR_ROI_CONFIGS = [
    {"max_evals": 300, "drift_1": 0.33, "drift_2": 0.56, "drift_3": 0.78},
    {"max_evals": 400, "drift_1": 0.33, "drift_2": 0.56, "drift_3": 0.78},
]
FOUR_ROI_ALL_QUADS = [
    ("V1v",  "OFA",   "FFA-1", "PPA"),
    ("EBA",  "OFA",   "FFA-1", "aTL-faces"),
    ("V1v",  "OPA",   "PPA",   "RSC"),
    ("OFA",  "FFA-1", "PPA",   "RSC"),
]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "--",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

BOOTSTRAP_N = 2_000   

def _pk(*rois):
    return "_".join(ROI_CAL_KEY[r] for r in rois)

def _seed_dirs(d: Path) -> List[Path]:
    return sorted(d.glob("seed_*")) if d.exists() else []

def _load_npy(p: Path) -> Optional[np.ndarray]:
    try:
        return np.load(p)
    except Exception:
        return None

def _collect(d: Path, fname: str) -> List[np.ndarray]:
    return [a for sd in _seed_dirs(d)
            for a in [_load_npy(sd / fname)]
            if a is not None and len(a) > 0]

def _load_json(p: Path) -> Optional[dict]:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

def _cd_two(roi_a, roi_b, algo, max_evals):
    return OUT_DIR / "two_roi" / _pk(roi_a, roi_b) / algo / f"evals_{max_evals}"

def _cd_three(trip, algo, max_evals):
    return OUT_DIR / "three_roi" / _pk(*trip) / algo / f"evals_{max_evals}"

def _cd_four(quad, algo, max_evals):
    return OUT_DIR / "four_roi" / _pk(*quad) / algo / f"evals_{max_evals}"

def _sep(title: str, width: int = 72):
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")

def _sub(title: str):
    print(f"\n  ── {title}")


def _cohens_d(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    sd = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return float((np.mean(a) - np.mean(b)) / sd) if sd > 1e-12 else float("nan")

def _bootstrap_d_ci(a, b, n=BOOTSTRAP_N, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ds = []
    for _ in range(n):
        sa = rng.choice(a, len(a), replace=True)
        sb = rng.choice(b, len(b), replace=True)
        ds.append(_cohens_d(sa, sb))
    ds = np.array(ds)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

def _welch(a, b):
    a = [x for x in (a or []) if np.isfinite(x)]
    b = [x for x in (b or []) if np.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(p), _cohens_d(a, b)

def _mannwhitney(a, b):
    a = [x for x in (a or []) if np.isfinite(x)]
    b = [x for x in (b or []) if np.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    _, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    u1, _ = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 1 - (2 * u1) / (n1 * n2)
    return float(p), float(r)

def _shapiro(x):
    x = [v for v in x if np.isfinite(v)]
    if len(x) < 3:
        return float("nan"), float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p = scipy_stats.shapiro(x)
    return float(stat), float(p)

def _levene(*groups):
    clean = [[v for v in g if np.isfinite(v)] for g in groups]
    clean = [g for g in clean if len(g) >= 2]
    if len(clean) < 2:
        return float("nan"), float("nan")
    stat, p = scipy_stats.levene(*clean)
    return float(stat), float(p)

def _sig(p):
    if np.isnan(p): return "   "
    if p < 0.001:   return "***"
    if p < 0.01:    return "** "
    if p < 0.05:    return "*  "
    return "ns "

def _fmt(vals):
    if not vals:
        return "n/a"
    return f"{np.mean(vals):>+.4f} ± {np.std(vals, ddof=1):.4f}  (n={len(vals)})"


def _integrity_condition(cond_dir: Path, max_evals: int,
                         raw_keys: List[str]) -> dict:
    issues   = []
    seed_dirs = _seed_dirs(cond_dir)
    n_seeds  = len(seed_dirs)

    length_ok  = 0
    nan_seeds  = []
    short_seeds = []
    missing_result = []
    log_eval_mismatch = []

    for sd in seed_dirs:
        if not (sd / "result.json").exists():
            missing_result.append(sd.name)

        for key in raw_keys:
            arr = _load_npy(sd / key)
            if arr is None:
                issues.append(f"{sd.name}/{key}: missing")
                continue
            if len(arr) < max_evals:
                short_seeds.append(f"{sd.name}/{key}: len={len(arr)}")
            else:
                length_ok += 1
            if not np.all(np.isfinite(arr)):
                nan_seeds.append(f"{sd.name}/{key}: "
                                 f"{np.sum(~np.isfinite(arr))} non-finite values")

        log = _load_json(sd / "drift_log.json")
        if log is not None:
            n_logged = len(log)
            if abs(n_logged - max_evals) > max_evals * 0.05: 
                log_eval_mismatch.append(f"{sd.name}: logged {n_logged}, "
                                         f"expected ~{max_evals}")

    return dict(
        n_seeds            = n_seeds,
        length_ok          = length_ok,
        short_seeds        = short_seeds,
        nan_seeds          = nan_seeds,
        missing_result     = missing_result,
        log_eval_mismatch  = log_eval_mismatch,
        other_issues       = issues,
    )


def check_integrity():
    _sep("§1  DATA INTEGRITY")

    specs = []
    for cfg in TWO_ROI_CONFIGS:
        e = cfg["max_evals"]
        for pair in TWO_ROI_ALL_PAIRS:
            for algo in ALGOS:
                specs.append((_cd_two(*pair, algo, e), e,
                               ["raw_a.npy", "raw_b.npy"],
                               f"2roi/{_pk(*pair)}/{ALGO_LABEL[algo]}/e{e}"))

    for cfg in THREE_ROI_CONFIGS:
        e = cfg["max_evals"]
        for trip in THREE_ROI_ALL_TRIPLETS:
            for algo in ALGOS:
                specs.append((_cd_three(trip, algo, e), e,
                               ["raw_a.npy", "raw_b.npy", "raw_c.npy"],
                               f"3roi/{_pk(*trip)}/{ALGO_LABEL[algo]}/e{e}"))

    for cfg in FOUR_ROI_CONFIGS:
        e = cfg["max_evals"]
        for quad in FOUR_ROI_ALL_QUADS:
            for algo in ALGOS:
                specs.append((_cd_four(quad, algo, e), e,
                               ["raw_a.npy","raw_b.npy","raw_c.npy","raw_d.npy"],
                               f"4roi/{_pk(*quad)}/{ALGO_LABEL[algo]}/e{e}"))

    total_conditions = len(specs)
    total_issues     = 0
    missing_dirs     = 0

    for d, max_evals, keys, label in specs:
        if not d.exists():
            missing_dirs += 1
            continue
        r = _integrity_condition(d, max_evals, keys)
        cond_issues = (len(r["short_seeds"]) + len(r["nan_seeds"]) +
                       len(r["missing_result"]) + len(r["log_eval_mismatch"]) +
                       len(r["other_issues"]))
        total_issues += cond_issues
        if cond_issues > 0 or r["n_seeds"] == 0:
            print(f"\n  [!] {label}")
            print(f"      seeds found : {r['n_seeds']}")
            if r["missing_result"]:
                print(f"      missing result.json : {r['missing_result']}")
            if r["short_seeds"]:
                print(f"      short arrays        : {r['short_seeds'][:5]}"
                      f"{'…' if len(r['short_seeds'])>5 else ''}")
            if r["nan_seeds"]:
                print(f"      non-finite values   : {r['nan_seeds'][:5]}"
                      f"{'…' if len(r['nan_seeds'])>5 else ''}")
            if r["log_eval_mismatch"]:
                print(f"      log count mismatch  : {r['log_eval_mismatch'][:5]}")
            if r["other_issues"]:
                print(f"      other               : {r['other_issues'][:5]}")

    print(f"\n  Summary: {total_conditions} conditions checked, "
          f"{missing_dirs} dirs missing, {total_issues} issues found.")
    if total_issues == 0 and missing_dirs == 0:
        print("  ✓  All conditions clean.")



def _load_weight_arrays(d: Path, w_keys: List[str]):
    out = {k: [] for k in w_keys}
    for sd in _seed_dirs(d):
        for k in w_keys:
            arr = _load_npy(sd / f"{k}.npy")
            if arr is not None and len(arr) > 0:
                out[k].append(arr)
    return out


def _plot_weight_trajectories(cond_dirs_labels, w_keys, transitions_frac,
                               max_evals, title, out_path):
    colors = ["#e67e22", "#8e44ad", "#2980b9", "#27ae60"]
    t      = np.arange(max_evals)

    fig, axes = plt.subplots(1, len(cond_dirs_labels),
                             figsize=(5 * len(cond_dirs_labels), 4),
                             sharey=True)
    if len(cond_dirs_labels) == 1:
        axes = [axes]

    for ax, (d, label) in zip(axes, cond_dirs_labels):
        wa = _load_weight_arrays(d, w_keys)
        for k, col in zip(w_keys, colors):
            arrs = wa[k]
            if not arrs:
                continue
            n   = min(len(a) for a in arrs)
            mat = np.stack([a[:n] for a in arrs if len(a) >= n])
            m   = mat.mean(0)
            s   = scipy_stats.sem(mat, axis=0)
            ax.plot(np.arange(n), m, color=col, lw=2, label=k)
            ax.fill_between(np.arange(n), m - s, m + s, color=col, alpha=0.15)
        for frac in transitions_frac:
            ax.axvline(frac * max_evals, color="black", ls=":", lw=1.5, alpha=0.6)
        ax.set_xlim(0, max_evals); ax.set_ylim(0, 1.05)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Eval")
        if axes.index(ax) == 0:
            ax.set_ylabel("ROI weight")
            ax.legend(fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def _check_weight_balance(d: Path, w_keys: List[str],
                          max_evals: int, transitions_frac: List[float],
                          label: str):
    wa = _load_weight_arrays(d, w_keys)
    issues = []
    for ki, k in enumerate(w_keys):
        arrs = wa[k]
        if not arrs:
            continue
        n   = min(len(a) for a in arrs)
        mat = np.stack([a[:n] for a in arrs if len(a) >= n])
        m   = mat.mean(0)
        bounds = ([0] + [int(f * max_evals) for f in transitions_frac]
                  + [max_evals])
        ph_start, ph_end = bounds[ki], bounds[ki + 1]
        outside = np.concatenate([m[:ph_start], m[ph_end:]])
        if len(outside) > 0 and outside.mean() > 0.70:
            issues.append(f"{k} dominates outside phase "
                          f"[{ph_start}:{ph_end}]  "
                          f"mean_outside={outside.mean():.2f}")
    if issues:
        print(f"  [!] {label}")
        for iss in issues:
            print(f"      {iss}")


def _check_per_seed_variance(d: Path, fname: str,
                              max_evals: int, label: str,
                              cv_threshold: float = 2.0):

    arrs = _collect(d, fname)
    if len(arrs) < 3:
        return
    tail  = max(1, max_evals // 10)
    finals = np.array([a[-tail:].mean() for a in arrs if len(a) >= max_evals])
    if len(finals) < 3:
        return
    z = np.abs(scipy_stats.zscore(finals))
    outliers = np.where(z > cv_threshold)[0]
    if len(outliers):
        print(f"  [!] Outlier seeds in {label}/{fname}:")
        for idx in outliers:
            print(f"      seed index {idx}: final_mean={finals[idx]:.4f}  "
                  f"z={z[idx]:.2f}")


def check_drift_sanity():
    _sep("§2  REWARD / DRIFT SANITY")

    _sub("2a  Weight trajectories")

    for cfg in TWO_ROI_CONFIGS:
        e = cfg["max_evals"]; mid = cfg["midpoint"]
        for pair in TWO_ROI_ALL_PAIRS:
            tag   = _pk(*pair)
            dirs  = [(_cd_two(*pair, algo, e), ALGO_LABEL[algo]) for algo in ALGOS]
            opath = DIAGDIR / f"weights_2roi_{tag}_e{e}.png"
            _plot_weight_trajectories(dirs, ["w_b"], [mid], e,
                                      f"Two-ROI {pair[0]}→{pair[1]}  e={e}", opath)

    for cfg in THREE_ROI_CONFIGS:
        e = cfg["max_evals"]; d1 = cfg["drift_1"]; d2 = cfg["drift_2"]
        for trip in THREE_ROI_ALL_TRIPLETS:
            tag  = _pk(*trip)
            dirs = [(_cd_three(trip, algo, e), ALGO_LABEL[algo]) for algo in ALGOS]
            opath = DIAGDIR / f"weights_3roi_{tag}_e{e}.png"
            _plot_weight_trajectories(dirs, ["w_a","w_b","w_c"], [d1, d2], e,
                                      f"Three-ROI {'→'.join(trip)}  e={e}", opath)

    for cfg in FOUR_ROI_CONFIGS:
        e  = cfg["max_evals"]
        d1 = cfg["drift_1"]; d2 = cfg["drift_2"]; d3 = cfg["drift_3"]
        for quad in FOUR_ROI_ALL_QUADS:
            tag  = _pk(*quad)
            dirs = [(_cd_four(quad, algo, e), ALGO_LABEL[algo]) for algo in ALGOS]
            opath = DIAGDIR / f"weights_4roi_{tag}_e{e}.png"
            _plot_weight_trajectories(dirs, ["w_a","w_b","w_c","w_d"],
                                      [d1, d2, d3], e,
                                      f"Four-ROI {'→'.join(quad)}  e={e}", opath)

    _sub("2b  Weight dominance outside designated phase")

    for cfg in TWO_ROI_CONFIGS:
        e = cfg["max_evals"]; mid = cfg["midpoint"]
        for pair in TWO_ROI_ALL_PAIRS:
            for algo in ALGOS:
                d = _cd_two(*pair, algo, e)
                _check_weight_balance(d, ["w_b"], e, [mid],
                                      f"2roi/{_pk(*pair)}/{ALGO_LABEL[algo]}/e{e}")

    for cfg in THREE_ROI_CONFIGS:
        e = cfg["max_evals"]; d1 = cfg["drift_1"]; d2 = cfg["drift_2"]
        for trip in THREE_ROI_ALL_TRIPLETS:
            for algo in ALGOS:
                d = _cd_three(trip, algo, e)
                _check_weight_balance(d, ["w_a","w_b","w_c"], e, [d1, d2],
                                      f"3roi/{_pk(*trip)}/{ALGO_LABEL[algo]}/e{e}")

    for cfg in FOUR_ROI_CONFIGS:
        e  = cfg["max_evals"]
        d1 = cfg["drift_1"]; d2 = cfg["drift_2"]; d3 = cfg["drift_3"]
        for quad in FOUR_ROI_ALL_QUADS:
            for algo in ALGOS:
                d = _cd_four(quad, algo, e)
                _check_weight_balance(d, ["w_a","w_b","w_c","w_d"], e,
                                      [d1, d2, d3],
                                      f"4roi/{_pk(*quad)}/{ALGO_LABEL[algo]}/e{e}")

    _sub("2c  Outlier seeds (final-window z-score > 2.0)")

    for cfg in TWO_ROI_CONFIGS:
        e = cfg["max_evals"]
        for pair in TWO_ROI_ALL_PAIRS:
            for algo in ALGOS:
                d = _cd_two(*pair, algo, e)
                for fname in ["raw_a.npy", "raw_b.npy"]:
                    _check_per_seed_variance(
                        d, fname, e,
                        f"2roi/{_pk(*pair)}/{ALGO_LABEL[algo]}/e{e}")

    for cfg in THREE_ROI_CONFIGS:
        e = cfg["max_evals"]
        for trip in THREE_ROI_ALL_TRIPLETS:
            for algo in ALGOS:
                d = _cd_three(trip, algo, e)
                for fname in ["raw_a.npy", "raw_b.npy", "raw_c.npy"]:
                    _check_per_seed_variance(
                        d, fname, e,
                        f"3roi/{_pk(*trip)}/{ALGO_LABEL[algo]}/e{e}")

    for cfg in FOUR_ROI_CONFIGS:
        e = cfg["max_evals"]
        for quad in FOUR_ROI_ALL_QUADS:
            for algo in ALGOS:
                d = _cd_four(quad, algo, e)
                for fname in ["raw_a.npy","raw_b.npy","raw_c.npy","raw_d.npy"]:
                    _check_per_seed_variance(
                        d, fname, e,
                        f"4roi/{_pk(*quad)}/{ALGO_LABEL[algo]}/e{e}")

    print("  Done.")


def _assumption_block(label: str,
                      data_sl: List[float],
                      data_sa: List[float],
                      data_ad: List[float]):
    """
    Run normality + variance tests; choose t-test or Mann-Whitney U accordingly.
    Prints a compact table row and returns whether normality holds.
    """
    groups = {STATELESS: data_sl, EXPLORER: data_sa, ADAPTIVE: data_ad}

    sw = {a: _shapiro(groups[a]) for a in ALGOS}
    normal = {a: (not np.isnan(sw[a][1])) and sw[a][1] >= 0.05 for a in ALGOS}
    all_normal = all(normal.values())

    lev_stat, lev_p = _levene(data_sl, data_sa, data_ad)

    print(f"\n  {label}")
    for a in ALGOS:
        stat, p = sw[a]
        flag = "✓" if normal[a] else "✗"
        print(f"    Shapiro {ALGO_LABEL[a]:<24}: W={stat:.3f}  p={p:.3f}  {flag}")
    flag_lev = "✓" if (not np.isnan(lev_p)) and lev_p >= 0.05 else "✗"
    print(f"    Levene (equal variance)      : stat={lev_stat:.3f}  "
          f"p={lev_p:.3f}  {flag_lev}")

    test_name = "Welch t" if all_normal else "Mann-Whitney U"
    test_fn   = _welch if all_normal else _mannwhitney
    effect_lbl = "d" if all_normal else "r"

    for algo, data_alt in [(EXPLORER, data_sa), (ADAPTIVE, data_ad)]:
        p, eff = test_fn(data_alt, data_sl)
        print(f"    {test_name} ({ALGO_LABEL[algo]} vs SL): "
              f"p={p:.4f} {_sig(p)}  {effect_lbl}={eff:+.3f}")

    return all_normal


def _gather_tail_scores(d: Path, fname: str,
                        max_evals: int, tail: int) -> List[float]:
    arrs = _collect(d, fname)
    return [float(a[max_evals - tail:max_evals].mean())
            for a in arrs if len(a) >= max_evals]


def check_statistical_assumptions():
    _sep("§3  STATISTICAL ASSUMPTIONS")
    tail_frac = 0.25 
    _sub("3a  Two-ROI — peak new-ROI (raw_b tail)")
    for cfg in TWO_ROI_CONFIGS:
        e    = cfg["max_evals"]
        tail = max(1, int(tail_frac * e))
        print(f"\n  [evals={e}]")
        for pair in TWO_ROI_ALL_PAIRS:
            data = {a: _gather_tail_scores(
                           _cd_two(*pair, a, e), "raw_b.npy", e, tail)
                    for a in ALGOS}
            _assumption_block(
                f"{pair[0]}→{pair[1]}",
                data[STATELESS], data[EXPLORER], data[ADAPTIVE])

    _sub("3b  Three-ROI — phase-2 tail (raw_c)")
    for cfg in THREE_ROI_CONFIGS:
        e    = cfg["max_evals"]
        T2   = int(cfg["drift_2"] * e)
        tail = max(1, (e - T2) // 2)
        print(f"\n  [evals={e}  T2={T2}]")
        for trip in THREE_ROI_ALL_TRIPLETS:
            data = {a: _gather_tail_scores(
                           _cd_three(trip, a, e), "raw_c.npy", e, tail)
                    for a in ALGOS}
            _assumption_block(
                "→".join(trip),
                data[STATELESS], data[EXPLORER], data[ADAPTIVE])

    _sub("3c  Four-ROI — phase-3 tail (raw_d)")
    for cfg in FOUR_ROI_CONFIGS:
        e    = cfg["max_evals"]
        T3   = int(cfg["drift_3"] * e)
        tail = max(1, (e - T3) // 2)
        print(f"\n  [evals={e}  T3={T3}]")
        for quad in FOUR_ROI_ALL_QUADS:
            data = {a: _gather_tail_scores(
                           _cd_four(quad, a, e), "raw_d.npy", e, tail)
                    for a in ALGOS}
            _assumption_block(
                "→".join(quad),
                data[STATELESS], data[EXPLORER], data[ADAPTIVE])


def _bootstrap_table(rows: List[Tuple], title: str):
    """
    rows : list of (label, data_sl, data_sa, data_ad)
    Prints a formatted table with point estimate + 95 % CI for Cohen's d.
    """
    rng = np.random.default_rng(42)
    W   = 72
    print(f"\n  {'Condition':<32}  {'algo':<10}  {'d':>6}  {'95 % CI':>18}  sig")
    print(f"  {'-'*W}")
    for label, sl, sa, ad in rows:
        for tag, alt in [("SA", sa), ("AD", ad)]:
            if len(sl) < 2 or len(alt) < 2:
                continue
            d         = _cohens_d(alt, sl)
            lo, hi    = _bootstrap_d_ci(alt, sl, rng=rng)
            _, p_w    = scipy_stats.ttest_ind(
                [x for x in alt if np.isfinite(x)],
                [x for x in sl  if np.isfinite(x)],
                equal_var=False)
            star = _sig(p_w)
            print(f"  {label:<32}  {tag:<10}  {d:>+6.3f}  "
                  f"[{lo:>+.3f}, {hi:>+.3f}]  {star}")
    print()


def check_effect_size_cis(quick: bool = False):
    global BOOTSTRAP_N
    if quick:
        BOOTSTRAP_N = 200
        print("  (quick mode: bootstrap N=200)")

    _sep("§4  EFFECT SIZE CIs  (bootstrap 95 %,  Cohen's d)")
    tail_frac = 0.25

    _sub("4a  Two-ROI — peak new-ROI (raw_b tail)")
    for cfg in TWO_ROI_CONFIGS:
        e    = cfg["max_evals"]
        tail = max(1, int(tail_frac * e))
        print(f"\n  [evals={e}]")
        rows = []
        for pair in TWO_ROI_ALL_PAIRS:
            data = {a: _gather_tail_scores(
                           _cd_two(*pair, a, e), "raw_b.npy", e, tail)
                    for a in ALGOS}
            rows.append((f"{pair[0]}→{pair[1]}",
                         data[STATELESS], data[EXPLORER], data[ADAPTIVE]))
        _bootstrap_table(rows, f"Two-ROI e={e}")

    _sub("4b  Three-ROI — phase-2 tail (raw_c)")
    for cfg in THREE_ROI_CONFIGS:
        e    = cfg["max_evals"]
        T2   = int(cfg["drift_2"] * e)
        tail = max(1, (e - T2) // 2)
        print(f"\n  [evals={e}  T2={T2}]")
        rows = []
        for trip in THREE_ROI_ALL_TRIPLETS:
            data = {a: _gather_tail_scores(
                           _cd_three(trip, a, e), "raw_c.npy", e, tail)
                    for a in ALGOS}
            rows.append(("→".join(trip),
                         data[STATELESS], data[EXPLORER], data[ADAPTIVE]))
        _bootstrap_table(rows, f"Three-ROI e={e}")

    _sub("4c  Four-ROI — phase-3 tail (raw_d)")
    for cfg in FOUR_ROI_CONFIGS:
        e    = cfg["max_evals"]
        T3   = int(cfg["drift_3"] * e)
        tail = max(1, (e - T3) // 2)
        print(f"\n  [evals={e}  T3={T3}]")
        rows = []
        for quad in FOUR_ROI_ALL_QUADS:
            data = {a: _gather_tail_scores(
                           _cd_four(quad, a, e), "raw_d.npy", e, tail)
                    for a in ALGOS}
            rows.append(("→".join(quad),
                         data[STATELESS], data[EXPLORER], data[ADAPTIVE]))
        _bootstrap_table(rows, f"Four-ROI e={e}")


def _direction_consistent(d_small: float, d_large: float) -> str:
    """Do both budgets show the same direction of effect?"""
    if np.isnan(d_small) or np.isnan(d_large):
        return "?"
    return "✓" if (d_small * d_large >= 0) else "✗ FLIP"


def check_budget_sensitivity():
    _sep("§5  BUDGET SENSITIVITY")
    tail_frac = 0.25

    def _d_for(d, fname, e, tail):
        data = {a: _gather_tail_scores(d(a), fname, e, tail) for a in ALGOS}
        p_sa, d_sa = _welch(data[EXPLORER], data[STATELESS])
        p_ad, d_ad = _welch(data[ADAPTIVE],  data[STATELESS])
        return d_sa, p_sa, d_ad, p_ad

    _sub("5a  Two-ROI — peak new-ROI (raw_b tail)")
    cfgs  = TWO_ROI_CONFIGS   # [e150, e200]
    e0, e1 = cfgs[0]["max_evals"], cfgs[1]["max_evals"]
    print(f"\n  {'Condition':<22}  {'algo':<4}  "
          f"{'d (e='+str(e0)+')':<12}  {'d (e='+str(e1)+')':<12}  consistent?")
    print(f"  {'-'*65}")
    for pair in TWO_ROI_ALL_PAIRS:
        for algo_tag, algo in [("SA", EXPLORER), ("AD", ADAPTIVE)]:
            def mk(e): return lambda a: _cd_two(*pair, a, e)
            t0 = max(1, int(tail_frac * e0)); t1 = max(1, int(tail_frac * e1))
            d0_sa, _, d0_ad, _ = _d_for(mk(e0), "raw_b.npy", e0, t0)
            d1_sa, _, d1_ad, _ = _d_for(mk(e1), "raw_b.npy", e1, t1)
            dA = d0_sa if algo_tag == "SA" else d0_ad
            dB = d1_sa if algo_tag == "SA" else d1_ad
            flag = _direction_consistent(dA, dB)
            print(f"  {pair[0]+'→'+pair[1]:<22}  {algo_tag:<4}  "
                  f"{dA:>+.3f}{'':6}  {dB:>+.3f}{'':6}  {flag}")

    _sub("5b  Three-ROI — phase-2 tail (raw_c)")
    cfgs  = THREE_ROI_CONFIGS
    e0, e1 = cfgs[0]["max_evals"], cfgs[1]["max_evals"]
    T2_0   = int(cfgs[0]["drift_2"] * e0)
    T2_1   = int(cfgs[1]["drift_2"] * e1)
    print(f"\n  {'Condition':<26}  {'algo':<4}  "
          f"{'d (e='+str(e0)+')':<12}  {'d (e='+str(e1)+')':<12}  consistent?")
    print(f"  {'-'*65}")
    for trip in THREE_ROI_ALL_TRIPLETS:
        for algo_tag, algo in [("SA", EXPLORER), ("AD", ADAPTIVE)]:
            def mk(e): return lambda a: _cd_three(trip, a, e)
            t0 = max(1, (e0 - T2_0) // 2); t1 = max(1, (e1 - T2_1) // 2)
            d0_sa, _, d0_ad, _ = _d_for(mk(e0), "raw_c.npy", e0, t0)
            d1_sa, _, d1_ad, _ = _d_for(mk(e1), "raw_c.npy", e1, t1)
            dA = d0_sa if algo_tag == "SA" else d0_ad
            dB = d1_sa if algo_tag == "SA" else d1_ad
            flag = _direction_consistent(dA, dB)
            print(f"  {'→'.join(trip):<26}  {algo_tag:<4}  "
                  f"{dA:>+.3f}{'':6}  {dB:>+.3f}{'':6}  {flag}")

    _sub("5c  Four-ROI — phase-3 tail (raw_d)")
    cfgs  = FOUR_ROI_CONFIGS
    e0, e1 = cfgs[0]["max_evals"], cfgs[1]["max_evals"]
    T3_0   = int(cfgs[0]["drift_3"] * e0)
    T3_1   = int(cfgs[1]["drift_3"] * e1)
    print(f"\n  {'Condition':<30}  {'algo':<4}  "
          f"{'d (e='+str(e0)+')':<12}  {'d (e='+str(e1)+')':<12}  consistent?")
    print(f"  {'-'*65}")
    for quad in FOUR_ROI_ALL_QUADS:
        for algo_tag, algo in [("SA", EXPLORER), ("AD", ADAPTIVE)]:
            def mk(e): return lambda a: _cd_four(quad, a, e)
            t0 = max(1, (e0 - T3_0) // 2); t1 = max(1, (e1 - T3_1) // 2)
            d0_sa, _, d0_ad, _ = _d_for(mk(e0), "raw_d.npy", e0, t0)
            d1_sa, _, d1_ad, _ = _d_for(mk(e1), "raw_d.npy", e1, t1)
            dA = d0_sa if algo_tag == "SA" else d0_ad
            dB = d1_sa if algo_tag == "SA" else d1_ad
            flag = _direction_consistent(dA, dB)
            print(f"  {'→'.join(quad):<30}  {algo_tag:<4}  "
                  f"{dA:>+.3f}{'':6}  {dB:>+.3f}{'':6}  {flag}")


def _shared_pairs() -> List[Tuple[str, str]]:
    seen: dict = {}

    for pair in TWO_ROI_ALL_PAIRS:
        key = (pair[0], pair[1])
        seen.setdefault(key, set()).add("2roi")

    for trip in THREE_ROI_ALL_TRIPLETS:
        for a, b in zip(trip, trip[1:]):
            seen.setdefault((a, b), set()).add("3roi")

    for quad in FOUR_ROI_ALL_QUADS:
        for a, b in zip(quad, quad[1:]):
            seen.setdefault((a, b), set()).add("4roi")

    return [(k, v) for k, v in seen.items() if len(v) > 1]


def check_cross_experiment():
    _sep("§6  CROSS-EXPERIMENT ROI CONSISTENCY")

    shared = _shared_pairs()
    if not shared:
        print("  No ROI pairs shared across experiment types.")
        return

    print(f"  Found {len(shared)} shared consecutive ROI pair(s):\n")
    tail_frac = 0.25

    for (roi_a, roi_b), exps in shared:
        print(f"  {'─'*60}")
        print(f"  {roi_a} → {roi_b}   appears in: {', '.join(sorted(exps))}")

        rows_sa = []; rows_ad = []

        if "2roi" in exps and (roi_a, roi_b) in TWO_ROI_ALL_PAIRS:
            for cfg in TWO_ROI_CONFIGS:
                e    = cfg["max_evals"]
                tail = max(1, int(tail_frac * e))
                data = {a: _gather_tail_scores(
                               _cd_two(roi_a, roi_b, a, e),
                               "raw_b.npy", e, tail)
                        for a in ALGOS}
                _, d_sa = _welch(data[EXPLORER], data[STATELESS])
                _, d_ad = _welch(data[ADAPTIVE],  data[STATELESS])
                rows_sa.append((f"2roi e={e}", d_sa))
                rows_ad.append((f"2roi e={e}", d_ad))

        if "3roi" in exps:
            for trip in THREE_ROI_ALL_TRIPLETS:
                pairs_in_trip = list(zip(trip, trip[1:]))
                if (roi_a, roi_b) not in pairs_in_trip:
                    continue
                slot = pairs_in_trip.index((roi_a, roi_b))
                fname = ["raw_b.npy", "raw_c.npy"][slot]
                for cfg in THREE_ROI_CONFIGS:
                    e    = cfg["max_evals"]
                    T_tr = [int(cfg["drift_1"] * e), int(cfg["drift_2"] * e)]
                    end  = T_tr[slot + 1] if slot + 1 < len(T_tr) else e
                    tail = max(1, (end - T_tr[slot]) // 2)
                    data = {a: _gather_tail_scores(
                                   _cd_three(trip, a, e), fname, e, tail)
                            for a in ALGOS}
                    _, d_sa = _welch(data[EXPLORER], data[STATELESS])
                    _, d_ad = _welch(data[ADAPTIVE],  data[STATELESS])
                    rows_sa.append((f"3roi {'→'.join(trip)} e={e}", d_sa))
                    rows_ad.append((f"3roi {'→'.join(trip)} e={e}", d_ad))

        if "4roi" in exps:
            for quad in FOUR_ROI_ALL_QUADS:
                pairs_in_quad = list(zip(quad, quad[1:]))
                if (roi_a, roi_b) not in pairs_in_quad:
                    continue
                slot  = pairs_in_quad.index((roi_a, roi_b))
                fname = ["raw_b.npy", "raw_c.npy", "raw_d.npy"][slot]
                for cfg in FOUR_ROI_CONFIGS:
                    e    = cfg["max_evals"]
                    T_tr = [int(cfg["drift_1"] * e),
                            int(cfg["drift_2"] * e),
                            int(cfg["drift_3"] * e)]
                    end  = T_tr[slot + 1] if slot + 1 < len(T_tr) else e
                    tail = max(1, (end - T_tr[slot]) // 2)
                    data = {a: _gather_tail_scores(
                                   _cd_four(quad, a, e), fname, e, tail)
                            for a in ALGOS}
                    _, d_sa = _welch(data[EXPLORER], data[STATELESS])
                    _, d_ad = _welch(data[ADAPTIVE],  data[STATELESS])
                    rows_sa.append((f"4roi {'→'.join(quad)} e={e}", d_ad))
                    rows_ad.append((f"4roi {'→'.join(quad)} e={e}", d_ad))

        print(f"  {'Context':<42}  {'d (SA)':<10}  {'d (AD)':<10}")
        for (lbl_sa, d_sa), (_, d_ad) in zip(rows_sa, rows_ad):
            flag_sa = ("↑" if d_sa > 0.2 else "↓" if d_sa < -0.2 else "~")
            flag_ad = ("↑" if d_ad > 0.2 else "↓" if d_ad < -0.2 else "~")
            print(f"  {lbl_sa:<42}  {d_sa:>+.3f} {flag_sa}    {d_ad:>+.3f} {flag_ad}")

        sa_dirs = [np.sign(d) for _, d in rows_sa if not np.isnan(d)]
        ad_dirs = [np.sign(d) for _, d in rows_ad if not np.isnan(d)]
        sa_ok   = len(set(sa_dirs)) <= 1
        ad_ok   = len(set(ad_dirs)) <= 1
        print(f"  SA consistent across contexts: {'✓' if sa_ok else '✗ mixed directions'}")
        print(f"  AD consistent across contexts: {'✓' if ad_ok else '✗ mixed directions'}")


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Quality checks and statistical diagnostics for ROI drift experiments.")
    ap.add_argument("--quick", action="store_true",
                    help="Use N=200 bootstrap resamples instead of 2000 (faster)")
    ap.add_argument("--skip-integrity",   action="store_true")
    ap.add_argument("--skip-drift",       action="store_true")
    ap.add_argument("--skip-assumptions", action="store_true")
    ap.add_argument("--skip-cis",         action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    ap.add_argument("--skip-cross",       action="store_true")
    return ap.parse_args()


def main():
    args = _parse_args()
    print(f"Diagnostics output → {DIAGDIR}")

    if not args.skip_integrity:
        check_integrity()

    if not args.skip_drift:
        check_drift_sanity()

    if not args.skip_assumptions:
        check_statistical_assumptions()

    if not args.skip_cis:
        check_effect_size_cis(quick=args.quick)

    if not args.skip_sensitivity:
        check_budget_sensitivity()

    if not args.skip_cross:
        check_cross_experiment()

    print(f"\n{'═'*72}")
    print(f"  All diagnostics complete.  Plots → {DIAGDIR}")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()