from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats

OUT_DIR = Path("./cloned/results")
FIGDIR  = OUT_DIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

STATELESS = "genetic_clip_stateless"
EXPLORER  = "genetic_clip_explorer"
ADAPTIVE  = "genetic_clip_adaptive_salience"
ALGOS     = [STATELESS, EXPLORER, ADAPTIVE]

ALGO_COLOR = {STATELESS: "#d6604d", EXPLORER: "#4dac26", ADAPTIVE: "#2166ac"}
ALGO_LABEL = {
    STATELESS: "Stateless",
    EXPLORER:  "Classic state-aware",
    ADAPTIVE:  "Adaptive state-aware",
}

ROI_COLOR_AB  = ["#e67e22", "#2980b9"]                              
ROI_COLORS_3  = ["#e67e22", "#8e44ad", "#2980b9"]                   
ROI_COLORS_4  = ["#e67e22", "#8e44ad", "#2980b9", "#27ae60"]       

ROI_CAL_KEY = {
    "V1v": "V1v", "PPA": "PPA", "FFA-1": "FFA-1", "EBA": "EBA",
    "OFA": "OFA", "OPA": "OPA", "FBA-1": "FBA-1", "RSC": "RSC",
    "aTL-faces": "aTL-faces",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.18, "grid.linestyle": "--",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

TWO_ROI_CONFIGS   = [{"max_evals": 150, "midpoint": 0.60},
                     {"max_evals": 200, "midpoint": 0.60}]
TWO_ROI_FOCUS = [
    ("PPA",  "OFA", "Anti-correlated\nr = −0.33"),
    ("V1v",  "PPA", "Near-zero\nr = +0.14"),
    ("OPA",  "PPA", "Highly correlated\nr = +0.92"),
]
TWO_ROI_ALL_PAIRS = [
    ("FFA-1", "PPA"), ("OFA", "FFA-1"), ("EBA", "OFA"),
    ("V1v",  "PPA"),  ("OPA", "PPA"),   ("FBA-1", "EBA"),
    ("EBA",  "PPA"),  ("V1v", "FFA-1"), ("PPA", "V1v"),
    ("PPA",  "OFA"),
]

THREE_ROI_CONFIGS   = [{"max_evals": 250, "drift_1": 0.40, "drift_2": 0.70},
                       {"max_evals": 300, "drift_1": 0.40, "drift_2": 0.70}]
THREE_ROI_ALL_TRIPLETS = [
    ("V1v",   "OFA",   "FFA-1"),
    ("FFA-1", "PPA",   "V1v"),
    ("OFA",   "FFA-1", "PPA"),
    ("OFA",   "FFA-1", "aTL-faces"),
    ("OPA",   "PPA",   "RSC"),
]
THREE_ROI_BEST = [
    ("OFA",   "FFA-1", "PPA",       "OFA→FFA-1→PPA"),
    ("V1v",   "OFA",   "FFA-1",     "V1v→OFA→FFA-1"),
    ("OPA",   "PPA",   "RSC",       "OPA→PPA→RSC"),
]
THREE_ROI_TRAJ = ("OFA", "FFA-1", "PPA")

FOUR_ROI_CONFIGS   = [{"max_evals": 300, "drift_1": 0.33, "drift_2": 0.56, "drift_3": 0.78},
                      {"max_evals": 400, "drift_1": 0.33, "drift_2": 0.56, "drift_3": 0.78}]
FOUR_ROI_ALL_QUADS = [
    ("V1v",  "OFA",   "FFA-1", "PPA"),
    ("EBA",  "OFA",   "FFA-1", "aTL-faces"),
    ("V1v",  "OPA",   "PPA",   "RSC"),
    ("OFA",  "FFA-1", "PPA",   "RSC"),
]
QUAD_PHASES  = ("OFA",  "FFA-1", "PPA",   "RSC")  
QUAD_SELFCAL = ("V1v",  "OFA",   "FFA-1", "PPA")  

def _pk(*rois):
    return "_".join(ROI_CAL_KEY[r] for r in rois)

def _seed_dirs(d: Path):
    return sorted(d.glob("seed_*")) if d.exists() else []

def _load_npy(p: Path):
    try:
        return np.load(p)
    except Exception:
        return None

def _collect(d: Path, fname: str):
    return [a for sd in _seed_dirs(d)
            for a in [_load_npy(sd / fname)]
            if a is not None and len(a) > 0]

def _cd_two(roi_a, roi_b, algo, max_evals):
    return OUT_DIR / "two_roi" / _pk(roi_a, roi_b) / algo / f"evals_{max_evals}"

def _cd_three(trip, algo, max_evals):
    return OUT_DIR / "three_roi" / _pk(*trip) / algo / f"evals_{max_evals}"

def _cd_four(quad, algo, max_evals):
    return OUT_DIR / "four_roi" / _pk(*quad) / algo / f"evals_{max_evals}"


def _mean_sem(arrs, n=None):
    if not arrs:
        return None, None
    if n is None:
        n = min(len(a) for a in arrs)
    mat = np.stack([a[:n] for a in arrs if len(a) >= n])
    return (mat.mean(0), scipy_stats.sem(mat, axis=0)) if len(mat) else (None, None)

def _shapiro_ok(x):
    """True if Shapiro-Wilk does not reject normality (p >= 0.05) or n < 3."""
    x = [v for v in x if np.isfinite(v)]
    if len(x) < 3:
        return True   # can't test; fall back to t-test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = scipy_stats.shapiro(x)
    return p >= 0.05

def _auto_test(a, b):
    a = [x for x in (a or []) if np.isfinite(x)]
    b = [x for x in (b or []) if np.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), "d"

    if _shapiro_ok(a) and _shapiro_ok(b):
        _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        sd   = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
        eff  = (np.mean(a) - np.mean(b)) / sd if sd > 1e-12 else float("nan")
        return float(p), float(eff), "d"
    else:
        _, p   = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        u1, _  = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        r      = 1 - (2 * u1) / (len(a) * len(b))
        return float(p), float(r), "r"

def _sig(p):
    if np.isnan(p): return "   "
    if p < 0.001:   return "***"
    if p < 0.01:    return "** "
    if p < 0.05:    return "*  "
    return "ns "

def _bracket(ax, x0, x1, y, p, eff=None, eff_label="d", dy=None):
    if dy is None:
        dy = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06
    s   = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns")
    lbl = s if eff is None else f"{s}  {eff_label}={eff:+.2f}"
    ax.plot([x0, x0, x1, x1], [y, y + dy * .35, y + dy * .35, y],
            color="black", lw=1, clip_on=False)
    ax.text((x0 + x1) / 2, y + dy * .4, lbl,
            ha="center", va="bottom", fontsize=8, clip_on=False,
            fontweight="bold" if s != "ns" else "normal",
            color="black" if s != "ns" else "#999")

def _bar_panel(ax, data, title, ylabel=None, y_from_min=False):
    means = [np.mean(data[a]) if data[a] else 0 for a in ALGOS]
    errs  = [scipy_stats.sem(data[a]) if len(data[a]) > 1 else 0 for a in ALGOS]
    ax.bar(range(3), means, 0.55, yerr=errs, capsize=5,
           color=[ALGO_COLOR[a] for a in ALGOS], alpha=0.88,
           error_kw={"elinewidth": 2})
    ax.axhline(0, color="black", lw=0.8)
    all_v = np.concatenate([data[a] for a in ALGOS if data[a]])
    if len(all_v):
        ymin = max(0, float(np.min(all_v)) - 0.02) if y_from_min else 0
        ymax = float(np.max(all_v) + max(errs))
        rng  = ymax - ymin
        ax.set_ylim(ymin, ymax + rng * 0.55)
    y_top = max(m + e for m, e in zip(means, errs))
    rng2  = ax.get_ylim()[1] - ax.get_ylim()[0]
    for j, algo in enumerate([EXPLORER, ADAPTIVE]):
        if data[algo] and data[STATELESS]:
            p, eff, eff_lbl = _auto_test(data[algo], data[STATELESS])
            _bracket(ax, 0, ALGOS.index(algo),
                     y_top + rng2 * .09 + rng2 * .15 * j,
                     p, eff, eff_lbl, dy=rng2 * .08)
    ax.set_xticks(range(3))
    ax.set_xticklabels([ALGO_LABEL[a].replace(" ", "\n") for a in ALGOS], fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    n = len(data[STATELESS])
    ax.text(0.98, 0.02, f"n={n}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#888")

def _mv(data, algo):
    return (f"{np.mean(data[algo]):>+.4f}(n={len(data[algo])})"
            if data[algo] else "n/a")

def _print_sep(label):
    W = 72
    print(f"\n{'━' * W}")
    print(f"  {label}")
    print(f"{'━' * W}")


def _adapt_quality(arrs_new, arrs_old, transition, win):
    out = []
    for b, a in zip(arrs_new, arrs_old):
        n  = min(len(a), len(b))
        pb = b[max(0, transition - win):transition]
        qb = b[transition:min(transition + win, n)]
        pa = a[max(0, transition - win):transition]
        qa = a[transition:min(transition + win, n)]
        if len(pb) >= 4 and len(qb) >= 4:
            out.append(float((qb.mean() - pb.mean()) - (qa.mean() - pa.mean())))
    return out

def _peak_window(arrs, start, end, tail=30):
    ws = max(start, end - tail)
    return [float(a[ws:end].mean()) for a in arrs if len(a) >= end]

def _crossover(arrs_old, arrs_new, min_eval=None, smooth=8):
    crossed, never = [], 0
    for a, b in zip(arrs_old, arrs_new):
        n  = min(len(a), len(b))
        ra = np.convolve(a[:n], np.ones(smooth) / smooth, mode="valid")
        rb = np.convolve(b[:n], np.ones(smooth) / smooth, mode="valid")
        idx = np.where(rb > ra)[0]
        if min_eval is not None:
            idx = idx[idx + smooth // 2 >= min_eval]
        if len(idx) > 0:
            crossed.append(float(idx[0] + smooth // 2))
        else:
            never += 1
    return crossed, never


def _two_params(max_evals, midpoint):
    return dict(
        max_evals  = max_evals,
        transition = int(midpoint * max_evals),
        win        = 50,
        tail       = 50,
    )


def fig_two_peak_late(max_evals, midpoint=0.60):
    p = _two_params(max_evals, midpoint)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    for ax, (roi_a, roi_b, regime) in zip(axes, TWO_ROI_FOCUS):
        data = {a: _peak_window(
                        _collect(_cd_two(roi_a, roi_b, a, max_evals), "raw_b.npy"),
                        p["max_evals"] - p["tail"], p["max_evals"])
                for a in ALGOS}
        _bar_panel(ax, data, regime,
                   ylabel=f"Mean raw new-ROI score (last {p['tail']} evals)"
                   if axes.tolist().index(ax) == 0 else None,
                   y_from_min=True)
    fig.suptitle(f"Two-ROI: did the algorithm find the new brain target?  "
                 f"[{max_evals} evals]\n"
                 "Higher = success. Mean ± SEM.",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = FIGDIR / f"two_roi_peak_late_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    _stats_two_peak_late(max_evals, p)


def _stats_two_peak_late(max_evals, p):
    _print_sep(f"Two-ROI | Peak-late | evals={max_evals}")
    for roi_a, roi_b, regime in TWO_ROI_FOCUS:
        data = {a: _peak_window(
                        _collect(_cd_two(roi_a, roi_b, a, max_evals), "raw_b.npy"),
                        p["max_evals"] - p["tail"], p["max_evals"])
                for a in ALGOS}
        p_sa, eff_sa, lbl_sa = _auto_test(data[EXPLORER], data[STATELESS])
        p_ad, eff_ad, lbl_ad = _auto_test(data[ADAPTIVE],  data[STATELESS])
        print(f"  {roi_a}→{roi_b} [{regime.split(chr(10))[0]}]")
        print(f"    SL={_mv(data,STATELESS)}  SA={_mv(data,EXPLORER)}  "
              f"AD={_mv(data,ADAPTIVE)}")
        print(f"    SA vs SL: {lbl_sa}={eff_sa:+.2f} {_sig(p_sa)}  "
              f"AD vs SL: {lbl_ad}={eff_ad:+.2f} {_sig(p_ad)}")



def fig_two_adapt_quality(max_evals, midpoint=0.60):
    p = _two_params(max_evals, midpoint)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    for ax, (roi_a, roi_b, regime) in zip(axes, TWO_ROI_FOCUS):
        arrs = {a: (_collect(_cd_two(roi_a, roi_b, a, max_evals), "raw_b.npy"),
                    _collect(_cd_two(roi_a, roi_b, a, max_evals), "raw_a.npy"))
                for a in ALGOS}
        data = {a: _adapt_quality(arrs[a][0], arrs[a][1], p["transition"], p["win"])
                for a in ALGOS}
        _bar_panel(ax, data, regime,
                   ylabel="Adapt quality (Δraw_b − Δraw_a)"
                   if axes.tolist().index(ax) == 0 else None)
    fig.suptitle(f"Two-ROI: adaptation quality at transition (eval {p['transition']})  "
                 f"[{max_evals} evals]\n"
                 "Positive = new ROI rising while old ROI fell. Mean ± SEM.",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = FIGDIR / f"two_roi_adapt_quality_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    _stats_two_adapt_quality(max_evals, p)


def _stats_two_adapt_quality(max_evals, p):
    _print_sep(f"Two-ROI | Adapt quality | evals={max_evals}")
    for roi_a, roi_b, regime in TWO_ROI_FOCUS:
        arrs = {a: (_collect(_cd_two(roi_a, roi_b, a, max_evals), "raw_b.npy"),
                    _collect(_cd_two(roi_a, roi_b, a, max_evals), "raw_a.npy"))
                for a in ALGOS}
        data = {a: _adapt_quality(arrs[a][0], arrs[a][1], p["transition"], p["win"])
                for a in ALGOS}
        p_sa, eff_sa, lbl_sa = _auto_test(data[EXPLORER], data[STATELESS])
        p_ad, eff_ad, lbl_ad = _auto_test(data[ADAPTIVE],  data[STATELESS])
        print(f"  {roi_a}→{roi_b} [{regime.split(chr(10))[0]}]")
        print(f"    SL={_mv(data,STATELESS)}  SA={_mv(data,EXPLORER)}  "
              f"AD={_mv(data,ADAPTIVE)}")
        print(f"    SA vs SL: {lbl_sa}={eff_sa:+.2f} {_sig(p_sa)}  "
              f"AD vs SL: {lbl_ad}={eff_ad:+.2f} {_sig(p_ad)}")



def fig_two_crossover(max_evals, midpoint=0.60):
    p = _two_params(max_evals, midpoint)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    for ax, (roi_a, roi_b, regime) in zip(axes, TWO_ROI_FOCUS):
        is_anti = "anti" in regime.lower()
        min_ev  = p["transition"] + 5 if is_anti else None
        crossed = {}; never = {}
        for algo in ALGOS:
            d  = _cd_two(roi_a, roi_b, algo, max_evals)
            aa = _collect(d, "raw_a.npy")
            bb = _collect(d, "raw_b.npy")
            n  = min(len(aa), len(bb))
            c, nv = _crossover(aa[:n], bb[:n], min_eval=min_ev)
            crossed[algo] = c; never[algo] = nv
        for pos, algo in enumerate(ALGOS):
            vals  = crossed[algo]
            color = ALGO_COLOR[algo]
            rng   = np.random.default_rng(42 + pos)
            if len(vals) >= 3:
                vp = ax.violinplot([vals], positions=[pos], widths=0.48,
                                   showmeans=True, showmedians=False, showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(color); pc.set_alpha(0.45)
                vp["cmeans"].set_color(color); vp["cmeans"].set_linewidth(2.5)
            elif vals:
                ax.hlines(np.mean(vals), pos - .2, pos + .2, color=color, lw=2.5)
            jitter = rng.uniform(-.13, .13, len(vals))
            ax.scatter([pos + j for j in jitter], vals,
                       color=color, alpha=0.5, s=18, zorder=3)
            nv = never[algo]
            if nv > 0:
                xs = [pos + rng.uniform(-.13, .13) for _ in range(nv)]
                ax.scatter(xs, [max_evals + 8] * nv, marker="x",
                           color=color, s=45, lw=1.8, zorder=4, clip_on=False)
                ax.text(pos, max_evals + 18, f"n={nv}\nnever",
                        ha="center", va="bottom", fontsize=7, color=color, style="italic")
        ax.axhline(p["transition"], color="black", ls="--", lw=1.5, alpha=0.7)
        ax.fill_between([-.5, 2.5], [0, 0], [p["transition"]] * 2,
                        color="#ffffcc", alpha=0.3, zorder=0)
        ax.set_xlim(-.5, 2.5); ax.set_ylim(0, max_evals + 40)
        ax.set_xticks(range(3))
        ax.set_xticklabels([ALGO_LABEL[a].replace(" ", "\n") for a in ALGOS], fontsize=9)
        ax.set_title(regime, fontsize=11, fontweight="bold")
        if axes.tolist().index(ax) == 0:
            ax.set_ylabel("Eval when new ROI overtook old ROI", fontsize=10)
            ax.text(p["transition"] + 2, p["transition"] + 3,
                    f"transition (eval {p['transition']})",
                    fontsize=7, color="#444", va="bottom")
        ax.text(2.38, p["transition"] / 2, "free ride\nzone",
                ha="right", va="center", fontsize=8, color="#a07800", style="italic")
    fig.suptitle(f"Two-ROI: when did the new ROI overtake the old one?  "
                 f"[{max_evals} evals]\n"
                 "Below dotted line = free ride.  × = never genuinely adapted.",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = FIGDIR / f"two_roi_crossover_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    _stats_two_crossover(max_evals, p)


def _stats_two_crossover(max_evals, p):
    _print_sep(f"Two-ROI | Crossover | evals={max_evals}")
    for roi_a, roi_b, regime in TWO_ROI_FOCUS:
        is_anti = "anti" in regime.lower()
        min_ev  = p["transition"] + 5 if is_anti else None
        print(f"  {roi_a}→{roi_b} [{regime.split(chr(10))[0]}]")
        for algo in ALGOS:
            d  = _cd_two(roi_a, roi_b, algo, max_evals)
            aa = _collect(d, "raw_a.npy"); bb = _collect(d, "raw_b.npy")
            n  = min(len(aa), len(bb))
            c, nv = _crossover(aa[:n], bb[:n], min_eval=min_ev)
            mean_c = np.mean(c) if c else float("nan")
            print(f"    {ALGO_LABEL[algo]:<24}: mean={mean_c:>6.0f}  "
                  f"crossed={len(c):>2}  never={nv:>2}")



def fig_two_trajectories(max_evals, midpoint=0.60):
    p  = _two_params(max_evals, midpoint)
    t  = np.arange(max_evals)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for row, (roi_a, roi_b, regime) in enumerate(TWO_ROI_FOCUS):
        m_sl_a, _ = _mean_sem(_collect(_cd_two(roi_a, roi_b, STATELESS, max_evals),
                                       "raw_a.npy"), max_evals)
        m_sl_b, _ = _mean_sem(_collect(_cd_two(roi_a, roi_b, STATELESS, max_evals),
                                       "raw_b.npy"), max_evals)
        for col, algo in enumerate(ALGOS):
            ax = axes[row, col]
            m_a, s_a = _mean_sem(_collect(_cd_two(roi_a, roi_b, algo, max_evals),
                                          "raw_a.npy"), max_evals)
            m_b, s_b = _mean_sem(_collect(_cd_two(roi_a, roi_b, algo, max_evals),
                                          "raw_b.npy"), max_evals)
            if m_a is None or m_b is None:
                continue
            if algo == STATELESS:
                ax.plot(t, m_a, color=ROI_COLOR_AB[0], lw=2, ls="--", label=roi_a)
                ax.fill_between(t, m_a - s_a, m_a + s_a, color=ROI_COLOR_AB[0], alpha=0.12)
                ax.plot(t, m_b, color=ROI_COLOR_AB[1], lw=2, ls="-",  label=roi_b)
                ax.fill_between(t, m_b - s_b, m_b + s_b, color=ROI_COLOR_AB[1], alpha=0.12)
                ax.set_ylim(bottom=0)
                if row == 0:
                    ax.legend(fontsize=7, frameon=True,
                              title="dashed=old  solid=new", title_fontsize=6)
                ax.set_ylabel("Raw score", fontsize=9)
            else:
                if m_sl_a is None or m_sl_b is None:
                    continue
                k  = np.ones(7) / 7
                db = np.convolve(m_b - m_sl_b, k, mode="same")
                da = np.convolve(m_a - m_sl_a, k, mode="same")
                ax.plot(t, db, color=ROI_COLOR_AB[1], lw=2, ls="-",  label=f"{roi_b}−SL")
                ax.fill_between(t, 0, db, where=db >= 0, color=ROI_COLOR_AB[1], alpha=0.13)
                ax.plot(t, da, color=ROI_COLOR_AB[0], lw=2, ls="--", label=f"{roi_a}−SL")
                ax.fill_between(t, da, 0, where=da <= 0, color=ROI_COLOR_AB[0], alpha=0.13)
                ax.axhline(0, color="black", lw=1, alpha=0.4)
                if row == 0 and col == 1:
                    ax.legend(fontsize=7, frameon=True)
                ax.set_ylabel("Δ vs stateless", fontsize=9)
            ax.axvline(p["transition"], color="black", ls=":", lw=1.5, alpha=0.5)
            ax.set_xlim(0, max_evals)
            if row == len(TWO_ROI_FOCUS) - 1:
                ax.set_xlabel("Oracle evaluation", fontsize=10)
            if col == 0:
                ax.text(-0.32, 0.5, regime, transform=ax.transAxes,
                        fontsize=9, fontweight="bold", va="center", rotation=90, color="#333")
            if row == 0:
                ttl = (f"{ALGO_LABEL[algo]}\n(reference)"
                       if algo == STATELESS else ALGO_LABEL[algo])
                ax.set_title(ttl, fontsize=11, fontweight="bold", color=ALGO_COLOR[algo])
    fig.text(0.5, 0.01,
             "Left: absolute scores (stateless reference).  "
             "Centre & right: difference vs stateless — "
             "blue above 0 = better new ROI, orange below 0 = better at releasing old ROI.",
             ha="center", fontsize=8.5, style="italic", color="#555")
    fig.suptitle(f"Two-ROI: ROI score trajectories — 3 regimes × 3 algorithms  "
                 f"[{max_evals} evals]\n"
                 "Mean across seeds. Difference columns smoothed (7-eval window).",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(h_pad=2.5, w_pad=1.5)
    out = FIGDIR / f"two_roi_trajectories_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _three_params(max_evals, drift_1, drift_2):
    return dict(
        max_evals = max_evals,
        T1        = int(drift_1 * max_evals),
        T2        = int(drift_2 * max_evals),
        win       = 50,
        tail      = 30,
    )

def fig_three_aggregate(max_evals, drift_1=0.40, drift_2=0.70):
    p   = _three_params(max_evals, drift_1, drift_2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (transition, new_k, old_k, t_label) in zip(axes, [
        (p["T1"], "raw_b.npy", "raw_a.npy", f"T1 — first transition  (eval {p['T1']})"),
        (p["T2"], "raw_c.npy", "raw_b.npy", f"T2 — second transition (eval {p['T2']})"),
    ]):
        agg = {a: [] for a in ALGOS}
        for trip in THREE_ROI_ALL_TRIPLETS:
            for a in ALGOS:
                d  = _cd_three(trip, a, max_evals)
                nb = _collect(d, new_k); ob = _collect(d, old_k)
                n  = min(len(nb), len(ob))
                agg[a] += _adapt_quality(nb[:n], ob[:n], transition, p["win"])
        _bar_panel(ax, agg, t_label,
                   ylabel="Adapt quality (Δraw_new − Δraw_old)"
                   if axes.tolist().index(ax) == 0 else None)
    fig.suptitle(f"Three-ROI: adaptation quality aggregated across all triplets  "
                 f"[{max_evals} evals]\n"
                 "State-aware advantage grows at T2.",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = FIGDIR / f"three_roi_aggregate_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    _stats_three_aggregate(max_evals, p)


def _stats_three_aggregate(max_evals, p):
    _print_sep(f"Three-ROI | Aggregate T1/T2 | evals={max_evals}")
    for transition, new_k, old_k, t_label in [
        (p["T1"], "raw_b.npy", "raw_a.npy", "T1"),
        (p["T2"], "raw_c.npy", "raw_b.npy", "T2"),
    ]:
        agg = {a: [] for a in ALGOS}
        for trip in THREE_ROI_ALL_TRIPLETS:
            for a in ALGOS:
                d  = _cd_three(trip, a, max_evals)
                nb = _collect(d, new_k); ob = _collect(d, old_k)
                n  = min(len(nb), len(ob))
                agg[a] += _adapt_quality(nb[:n], ob[:n], transition, p["win"])
        p_sa, eff_sa, lbl_sa = _auto_test(agg[EXPLORER], agg[STATELESS])
        p_ad, eff_ad, lbl_ad = _auto_test(agg[ADAPTIVE],  agg[STATELESS])
        print(f"\n  {t_label}: SL={_mv(agg,STATELESS)}  SA={_mv(agg,EXPLORER)}  "
              f"AD={_mv(agg,ADAPTIVE)}")
        print(f"    SA vs SL: {lbl_sa}={eff_sa:+.2f} {_sig(p_sa)}  "
              f"AD vs SL: {lbl_ad}={eff_ad:+.2f} {_sig(p_ad)}")



def fig_three_peak_ph2(max_evals, drift_1=0.40, drift_2=0.70):
    p   = _three_params(max_evals, drift_1, drift_2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    for ax, (roi_a, roi_b, roi_c, label) in zip(axes, THREE_ROI_BEST):
        data = {a: _peak_window(
                        _collect(_cd_three((roi_a, roi_b, roi_c), a, max_evals),
                                 "raw_c.npy"),
                        p["T2"], p["max_evals"], p["tail"])
                for a in ALGOS}
        _bar_panel(ax, data, label,
                   ylabel=f"Mean raw activation of 2nd new target\n(last {p['tail']} evals)"
                   if axes.tolist().index(ax) == 0 else None,
                   y_from_min=True)
    fig.suptitle(f"Three-ROI: peak activation after T2 — did the algorithm find the "
                 f"third target?  [{max_evals} evals]\n"
                 "Higher = successfully driving the final target region. Mean ± SEM.",
                 fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = FIGDIR / f"three_roi_peak_ph2_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    _stats_three_peak_ph2(max_evals, p)


def _stats_three_peak_ph2(max_evals, p):
    _print_sep(f"Three-ROI | Peak phase 2 | evals={max_evals}")
    for roi_a, roi_b, roi_c, label in THREE_ROI_BEST:
        data = {a: _peak_window(
                        _collect(_cd_three((roi_a, roi_b, roi_c), a, max_evals),
                                 "raw_c.npy"),
                        p["T2"], p["max_evals"], p["tail"])
                for a in ALGOS}
        p_sa, eff_sa, lbl_sa = _auto_test(data[EXPLORER], data[STATELESS])
        p_ad, eff_ad, lbl_ad = _auto_test(data[ADAPTIVE],  data[STATELESS])
        print(f"\n  {label}")
        print(f"    SL={_mv(data,STATELESS)}  SA={_mv(data,EXPLORER)}  "
              f"AD={_mv(data,ADAPTIVE)}")
        print(f"    SA vs SL: {lbl_sa}={eff_sa:+.2f} {_sig(p_sa)}  "
              f"AD vs SL: {lbl_ad}={eff_ad:+.2f} {_sig(p_ad)}")



def fig_three_trajectories(max_evals, drift_1=0.40, drift_2=0.70):
    p    = _three_params(max_evals, drift_1, drift_2)
    trip = THREE_ROI_TRAJ
    t    = np.arange(max_evals)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, algo in zip(axes, ALGOS):
        d = _cd_three(trip, algo, max_evals)
        for fname, roi_name, color, ls in [
            ("raw_a.npy", trip[0], ROI_COLORS_3[0], "--"),
            ("raw_b.npy", trip[1], ROI_COLORS_3[1], "-."),
            ("raw_c.npy", trip[2], ROI_COLORS_3[2], "-"),
        ]:
            m, s = _mean_sem(_collect(d, fname), max_evals)
            if m is None:
                continue
            ax.plot(t, m, color=color, lw=2, ls=ls, label=roi_name)
            ax.fill_between(t, m - s, m + s, color=color, alpha=0.1)
        for tr in [p["T1"], p["T2"]]:
            ax.axvline(tr, color="#555", ls=":", lw=1.5, alpha=0.6)
        ax.set_xlim(0, max_evals); ax.set_ylim(bottom=0)
        ax.set_xlabel("Oracle evaluation", fontsize=10)
        ax.set_title(ALGO_LABEL[algo], fontsize=11,
                     fontweight="bold", color=ALGO_COLOR[algo])
        if axes.tolist().index(ax) == 0:
            ax.set_ylabel("Raw brain score", fontsize=10)
            ax.legend(fontsize=8, frameon=True)
    handles = [Line2D([0], [0], color="#555", ls=":", lw=1.5,
                      label=f"T1 (eval {p['T1']}) / T2 (eval {p['T2']})")]
    fig.legend(handles=handles, loc="lower center", fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle(f"Three-ROI: {'→'.join(trip)} — raw activation across two transitions  "
                 f"[{max_evals} evals]\n"
                 "Mean ± SEM.",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIGDIR / f"three_roi_trajectories_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _four_params(max_evals, drift_1, drift_2, drift_3):
    return dict(
        max_evals = max_evals,
        T1        = int(drift_1 * max_evals),
        T2        = int(drift_2 * max_evals),
        T3        = int(drift_3 * max_evals),
        win       = 40,
        tail      = 30,
    )

def _four_phases(p):
    return [
        (0,       p["T1"], "raw_a.npy", f"Phase 0\n(before T1)"),
        (p["T1"], p["T2"], "raw_b.npy", f"Phase 1\n(T1 → T2)"),
        (p["T2"], p["T3"], "raw_c.npy", f"Phase 2\n(T2 → T3)"),
        (p["T3"], p["max_evals"], "raw_d.npy", f"Phase 3\n(after T3)"),
    ]


def fig_four_phases(max_evals, drift_1=0.33, drift_2=0.56, drift_3=0.78):
    p       = _four_params(max_evals, drift_1, drift_2, drift_3)
    phases  = _four_phases(p)
    quad    = QUAD_PHASES
    targets = list(quad)

    fig, axes = plt.subplots(1, 4, figsize=(16, 7.5), sharey=False)
    fig.subplots_adjust(top=0.62, bottom=0.16, wspace=0.40)

    for ph_idx, (ph_start, ph_end, fname, ph_label) in enumerate(phases):
        ax   = axes[ph_idx]
        data = {a: _peak_window(_collect(_cd_four(quad, a, max_evals), fname),
                                ph_start, ph_end, p["tail"])
                for a in ALGOS}
        means = [np.mean(data[a]) if data[a] else 0 for a in ALGOS]
        errs  = [scipy_stats.sem(data[a]) if len(data[a]) > 1 else 0 for a in ALGOS]
        all_v = np.concatenate([data[a] for a in ALGOS if data[a]])
        if len(all_v):
            ymin = max(0, float(np.min(all_v)) - 0.03)
            ymax = float(np.max(all_v) + max(errs))
            rng  = ymax - ymin
            ax.set_ylim(ymin, ymax + rng * 0.50)
        ax.bar(range(3), means, 0.55, yerr=errs, capsize=5,
               color=[ALGO_COLOR[a] for a in ALGOS], alpha=0.88,
               error_kw={"elinewidth": 2})
        ax.axhline(0, color="black", lw=0.8, alpha=0.4)
        y_top = max(m + e for m, e in zip(means, errs))
        rng2  = ax.get_ylim()[1] - ax.get_ylim()[0]
        for j, algo in enumerate([EXPLORER, ADAPTIVE]):
            if data[algo] and data[STATELESS]:
                pv, eff, eff_lbl = _auto_test(data[algo], data[STATELESS])
                _bracket(ax, 0, ALGOS.index(algo),
                         y_top + rng2 * .09 + rng2 * .15 * j,
                         pv, eff, eff_lbl, dy=rng2 * .08)
        ax.set_xticks(range(3))
        ax.set_xticklabels([ALGO_LABEL[a].replace(" ", "\n") for a in ALGOS], fontsize=9)
        ax.set_title(ph_label, fontsize=11, fontweight="bold", pad=8)
        ax.text(0.5, -0.22, f"target: {targets[ph_idx]}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, color="#555", style="italic", fontweight="bold")
        if ph_idx == 0:
            ax.set_ylabel(f"Mean raw brain score\n(last {p['tail']} evals of phase)",
                          fontsize=10)
        if ph_idx == 2:
            arrs_b = _collect(_cd_four(quad, STATELESS, max_evals), "raw_b.npy")
            arrs_c = _collect(_cd_four(quad, STATELESS, max_evals), "raw_c.npy")
            n      = min(len(arrs_b), len(arrs_c))
            _, nv  = _crossover(arrs_b[:n], arrs_c[:n], min_eval=p["T2"])
            if nv > 0 and means[0] > 0:
                ax.text(0, means[0], f" {nv}/{len(arrs_b[:n])}\nnever\nadapted",
                        ha="left", va="top", fontsize=7.5,
                        color=ALGO_COLOR[STATELESS], fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor=ALGO_COLOR[STATELESS], alpha=0.85))

    fig.text(0.5, 0.98, f"Four-ROI: {'→'.join(quad)}  [{max_evals} evals]",
             ha="center", va="top", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.92,
             f"{quad[0]} ──T1({p['T1']})──► {quad[1]} ──T2({p['T2']})──► "
             f"{quad[2]} ──T3({p['T3']})──► {quad[3]}",
             ha="center", va="top", fontsize=10, color="#444")
    fig.text(0.5, 0.87,
             "Mean activation of active target ROI per phase — higher = algorithm found the target",
             ha="center", va="top", fontsize=9.5, style="italic", color="#666")
    handles = [mpatches.Patch(color=ALGO_COLOR[a], alpha=0.88, label=ALGO_LABEL[a])
               for a in ALGOS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, 0.01), frameon=True)
    out = FIGDIR / f"four_roi_phases_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    _stats_four_phases(max_evals, p)


def _stats_four_phases(max_evals, p):
    _print_sep(f"Four-ROI | Peak per phase | evals={max_evals}")
    quad    = QUAD_PHASES
    targets = list(quad)
    phases  = _four_phases(p)
    for ph_idx, (ph_start, ph_end, fname, ph_label) in enumerate(phases):
        data = {a: _peak_window(_collect(_cd_four(quad, a, max_evals), fname),
                                ph_start, ph_end, p["tail"])
                for a in ALGOS}
        p_sa, eff_sa, lbl_sa = _auto_test(data[EXPLORER], data[STATELESS])
        p_ad, eff_ad, lbl_ad = _auto_test(data[ADAPTIVE],  data[STATELESS])
        print(f"\n  {ph_label.replace(chr(10),' '):<22} target={targets[ph_idx]}")
        print(f"    SL={_mv(data,STATELESS)}  SA={_mv(data,EXPLORER)}  "
              f"AD={_mv(data,ADAPTIVE)}")
        print(f"    SA vs SL: {lbl_sa}={eff_sa:+.2f} {_sig(p_sa)}  "
              f"AD vs SL: {lbl_ad}={eff_ad:+.2f} {_sig(p_ad)}")
        if ph_idx > 0:
            prev_fname = phases[ph_idx - 1][2]
            prev_trans = [p["T1"], p["T2"], p["T3"]][ph_idx - 1]
            for a in ALGOS:
                ap = _collect(_cd_four(quad, a, max_evals), prev_fname)
                ac = _collect(_cd_four(quad, a, max_evals), fname)
                n  = min(len(ap), len(ac))
                cr, nv = _crossover(ap[:n], ac[:n], min_eval=prev_trans)
                print(f"    crossover {ALGO_LABEL[a]:<24}: "
                      f"crossed={cr}  never={nv}")



def fig_four_selfcal(max_evals, drift_1=0.33, drift_2=0.56, drift_3=0.78):
    p = _four_params(max_evals, drift_1, drift_2, drift_3)

    def _impr(quad):
        out = {a: [] for a in ALGOS}
        for a in ALGOS:
            d  = _cd_four(quad, a, max_evals)
            q1 = _adapt_quality(_collect(d, "raw_b.npy"),
                                 _collect(d, "raw_a.npy"), p["T1"], p["win"])
            q2 = _adapt_quality(_collect(d, "raw_c.npy"),
                                 _collect(d, "raw_b.npy"), p["T2"], p["win"])
            n  = min(len(q1), len(q2))
            out[a] = [q2[i] - q1[i] for i in range(n)]
        return out

    panels = [
        (_impr(QUAD_SELFCAL), f"Single quad ({' → '.join(QUAD_SELFCAL)})"),
        ({a: sum([_impr(q)[a] for q in FOUR_ROI_ALL_QUADS], []) for a in ALGOS},
         "Aggregate (all 4 quadruplets)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(top=0.80, bottom=0.12, wspace=0.35)

    for ax, (vals, title) in zip(axes, panels):
        means = [np.mean(vals[a]) if vals[a] else 0 for a in ALGOS]
        errs  = [scipy_stats.sem(vals[a]) if len(vals[a]) > 1 else 0 for a in ALGOS]
        ax.bar(range(3), means, 0.55, yerr=errs, capsize=6,
               color=[ALGO_COLOR[a] for a in ALGOS], alpha=0.88,
               error_kw={"elinewidth": 2.5})
        ax.axhline(0, color="black", lw=1.2)
        ymin, ymax = ax.get_ylim()
        ax.fill_between([-0.5, 2.5], [0, 0], [ymax, ymax], color="#d5f5e3", alpha=0.18, zorder=0)
        ax.fill_between([-0.5, 2.5], [ymin, ymin], [0, 0], color="#fdedec", alpha=0.18, zorder=0)
        ax.set_xlim(-0.5, 2.5)
        rng  = ymax - ymin
        ax.set_ylim(ymin - rng * .05, ymax + rng * .60)
        y_top = max(m + e for m, e in zip(means, errs))
        rng2  = ax.get_ylim()[1] - ax.get_ylim()[0]
        for j, algo in enumerate([EXPLORER, ADAPTIVE]):
            if vals[algo] and vals[STATELESS]:
                pv, eff, eff_lbl = _auto_test(vals[algo], vals[STATELESS])
                _bracket(ax, 0, ALGOS.index(algo),
                         y_top + rng2 * .10 + rng2 * .16 * j,
                         pv, eff, eff_lbl, dy=rng2 * .09)
        ax.set_xticks(range(3))
        ax.set_xticklabels([ALGO_LABEL[a] for a in ALGOS], fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        if axes.tolist().index(ax) == 0:
            ax.set_ylabel("T2 − T1 improvement\n(adapt quality at T2 minus T1)", fontsize=10)
        ax.text(0.04, 0.97, "improves at T2 ↑",
                transform=ax.transAxes, fontsize=8.5, va="top", color="#27ae60", style="italic")
        ax.text(0.04, 0.03, "degrades at T2 ↓",
                transform=ax.transAxes, fontsize=8.5, va="bottom", color="#e74c3c", style="italic")
        ax.text(0.98, 0.03, f"n={len(vals[STATELESS])}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#888")

    fig.text(0.5, 0.93,
             f"Four-ROI self-calibration: does the algorithm improve at the second transition?  "
             f"[{max_evals} evals]",
             ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(0.5, 0.87,
             "Positive = improved at T2 vs T1 — the adaptive window self-calibrates over the session.",
             ha="center", va="top", fontsize=9.5, style="italic", color="#555")
    out = FIGDIR / f"four_roi_selfcal_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    _stats_four_selfcal(max_evals, p)


def _stats_four_selfcal(max_evals, p):
    _print_sep(f"Four-ROI | Self-calibration T2−T1 | evals={max_evals}")
    for quad, label in [(QUAD_SELFCAL, "Single quad"),
                        (None, "Aggregate all quads")]:
        quads = FOUR_ROI_ALL_QUADS if quad is None else [quad]
        agg   = {a: [] for a in ALGOS}
        for q in quads:
            for a in ALGOS:
                d  = _cd_four(q, a, max_evals)
                q1 = _adapt_quality(_collect(d, "raw_b.npy"),
                                     _collect(d, "raw_a.npy"), p["T1"], p["win"])
                q2 = _adapt_quality(_collect(d, "raw_c.npy"),
                                     _collect(d, "raw_b.npy"), p["T2"], p["win"])
                n  = min(len(q1), len(q2))
                agg[a] += [q2[i] - q1[i] for i in range(n)]
        p_sa, eff_sa, lbl_sa = _auto_test(agg[EXPLORER], agg[STATELESS])
        p_ad, eff_ad, lbl_ad = _auto_test(agg[ADAPTIVE],  agg[STATELESS])
        print(f"\n  {label}")
        print(f"    SL={_mv(agg,STATELESS)}  SA={_mv(agg,EXPLORER)}  "
              f"AD={_mv(agg,ADAPTIVE)}")
        print(f"    SA vs SL: {lbl_sa}={eff_sa:+.2f} {_sig(p_sa)}  "
              f"AD vs SL: {lbl_ad}={eff_ad:+.2f} {_sig(p_ad)}")


def fig_four_trajectories(max_evals, drift_1=0.33, drift_2=0.56, drift_3=0.78):
    p      = _four_params(max_evals, drift_1, drift_2, drift_3)
    quads  = [QUAD_SELFCAL, QUAD_PHASES]
    labels = ["→".join(QUAD_SELFCAL), "→".join(QUAD_PHASES)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for row, (quad, qlabel) in enumerate(zip(quads, labels)):
        for col, algo in enumerate(ALGOS):
            ax = axes[row, col]
            for fi, (fname, color, ls) in enumerate(zip(
                ["raw_a.npy", "raw_b.npy", "raw_c.npy", "raw_d.npy"],
                ROI_COLORS_4,
                ["--", "-.", (0, (5, 2)), ":"],
            )):
                m, s = _mean_sem(_collect(_cd_four(quad, algo, max_evals), fname),
                                 max_evals)
                if m is None:
                    continue
                ax.plot(np.arange(max_evals), m, color=color, lw=2,
                        ls=ls, label=quad[fi])
                ax.fill_between(np.arange(max_evals), m - s, m + s,
                                color=color, alpha=0.1)
            for tr in [p["T1"], p["T2"], p["T3"]]:
                ax.axvline(tr, color="black", ls=":", lw=1.5, alpha=0.5)
            ax.set_xlim(0, max_evals); ax.set_ylim(bottom=0)
            if row == 1:
                ax.set_xlabel("Oracle evaluation", fontsize=10)
            if col == 0:
                ax.set_ylabel("Raw brain score", fontsize=9)
                ax.text(-0.30, 0.5, qlabel, transform=ax.transAxes,
                        fontsize=8, fontweight="bold", va="center", rotation=90)
            if row == 0:
                ax.set_title(ALGO_LABEL[algo], fontsize=11,
                             fontweight="bold", color=ALGO_COLOR[algo])
            if row == 0 and col == 0:
                ax.legend(fontsize=7, frameon=True)

    handles = [Line2D([0], [0], color="black", ls=":", lw=1.5,
                      label=f"T1 ({p['T1']}) / T2 ({p['T2']}) / T3 ({p['T3']})")]
    fig.legend(handles=handles, loc="lower center", fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=True)
    fig.suptitle(f"Four-ROI: raw activation across three sequential transitions  "
                 f"[{max_evals} evals]\n"
                 "Mean ± SEM.",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout(h_pad=2.5, w_pad=1.5)
    out = FIGDIR / f"four_roi_trajectories_e{max_evals}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Generate all figures for two/three/four-ROI drift experiments.")
    ap.add_argument("--evals2", type=int, default=200,
                    choices=[c["max_evals"] for c in TWO_ROI_CONFIGS],
                    help="Budget to plot for two-ROI (default: 200)")
    ap.add_argument("--evals3", type=int, default=300,
                    choices=[c["max_evals"] for c in THREE_ROI_CONFIGS],
                    help="Budget to plot for three-ROI (default: 300)")
    ap.add_argument("--evals4", type=int, default=400,
                    choices=[c["max_evals"] for c in FOUR_ROI_CONFIGS],
                    help="Budget to plot for four-ROI (default: 400)")
    ap.add_argument("--all-budgets", action="store_true",
                    help="Run every budget for every experiment (overrides --evalsN)")
    return ap.parse_args()


def main():
    np.random.seed(0)
    args = _parse_args()
    print(f"Output → {FIGDIR}\n")

    if args.all_budgets:
        two_cfgs   = TWO_ROI_CONFIGS
        three_cfgs = THREE_ROI_CONFIGS
        four_cfgs  = FOUR_ROI_CONFIGS
    else:
        two_cfgs   = [c for c in TWO_ROI_CONFIGS   if c["max_evals"] == args.evals2]
        three_cfgs = [c for c in THREE_ROI_CONFIGS if c["max_evals"] == args.evals3]
        four_cfgs  = [c for c in FOUR_ROI_CONFIGS  if c["max_evals"] == args.evals4]

    print("=" * 65)
    print("  TWO-ROI")
    print("=" * 65)
    for cfg in two_cfgs:
        e = cfg["max_evals"]; m = cfg["midpoint"]
        print(f"\n  evals={e}  midpoint={m}")
        print("  1/4  Peak new-ROI late …")
        fig_two_peak_late(e, m)
        print("  2/4  Adapt quality …")
        fig_two_adapt_quality(e, m)
        print("  3/4  Crossover …")
        fig_two_crossover(e, m)
        print("  4/4  Trajectories …")
        fig_two_trajectories(e, m)

    print("\n" + "=" * 65)
    print("  THREE-ROI")
    print("=" * 65)
    for cfg in three_cfgs:
        e = cfg["max_evals"]; d1 = cfg["drift_1"]; d2 = cfg["drift_2"]
        print(f"\n  evals={e}  drift_1={d1}  drift_2={d2}")
        print("  1/3  Aggregate T1 vs T2 …")
        fig_three_aggregate(e, d1, d2)
        print("  2/3  Peak phase 2 …")
        fig_three_peak_ph2(e, d1, d2)
        print("  3/3  Trajectories …")
        fig_three_trajectories(e, d1, d2)

    print("\n" + "=" * 65)
    print("  FOUR-ROI")
    print("=" * 65)
    for cfg in four_cfgs:
        e = cfg["max_evals"]; d1 = cfg["drift_1"]; d2 = cfg["drift_2"]; d3 = cfg["drift_3"]
        print(f"\n  evals={e}  drift_1={d1}  drift_2={d2}  drift_3={d3}")
        print("  1/3  Peak per phase …")
        fig_four_phases(e, d1, d2, d3)
        print("  2/3  Self-calibration …")
        fig_four_selfcal(e, d1, d2, d3)
        print("  3/3  Trajectories …")
        fig_four_trajectories(e, d1, d2, d3)

    print(f"\nAll done → {FIGDIR}")


if __name__ == "__main__":
    main()