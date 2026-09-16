"""Generate every figure in the dissertation from validation/chapter6_stats.json.

Figures are written to dissertation/figures/ as both PDF (for print submission)
and PNG (for review). No figure carries a number that is not also referenced in
the text, and no figure restates a table without adding something the table
cannot show: an interval, a proportion, or a flow.

Run: .venv/bin/python scripts/build_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

STATS = json.load(open("validation/chapter6_stats.json"))
OUT = Path("dissertation/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Print-safe greys with a single accent. A dissertation may be marked from a
# monochrome printout, so nothing is distinguished by hue alone.
INK = "#1a1a1a"
MID = "#6b6b6b"
LIGHT = "#c8c8c8"
ACCENT = "#8c2d2d"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def fig_funnel() -> None:
    """Figure 6.1 - acquisition funnel, log-scaled because the range is 5 orders."""
    stages = [
        ("AndroZoo index snapshot", 27_589_444),
        ("AI-candidate rows", 68_419),
        ("Unique applications", 16_364),
        ("Eligible 2024-2026", 6_333),
        ("Sampled", 1_500),
        ("Scanned", 1_495),
        ("Confirmed LLM-integrated", 111),
    ]
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = range(len(stages))
    colours = [LIGHT] * (len(stages) - 1) + [ACCENT]
    ax.barh(list(y), values, color=colours, edgecolor=INK, linewidth=0.6, height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Applications (log scale)")
    ax.set_xlim(50, 1e8)
    for i, v in enumerate(values):
        ax.text(v * 1.35, i, f"{v:,}", va="center", fontsize=8,
                color=ACCENT if i == len(values) - 1 else INK)
    ax.grid(axis="x", color=LIGHT, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig6-1-acquisition-funnel")


def fig_prevalence() -> None:
    """Figure 6.2 - prevalence with Wilson intervals.

    A forest plot rather than a bar chart, because the interval is the point:
    a bar chart of point estimates would hide exactly what Chapter 6 insists on
    reporting.
    """
    p = STATS["prevalence"]
    rows = [
        ("Any RQ1 artefact", "any_rq1_artefact", True),
        ("Inference-provider credential", "inference_credential", False),
        ("Confidential system prompt", "system_prompt", False),
        ("Retrieval corpus", "rag_artefact", False),
        ("Tool / function schema", "tool_schema", False),
        ("User-selectable preset", "preset_only", False),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, (label, key, headline) in enumerate(rows):
        d = p[key]
        lo, hi = d["wilson_lo"] * 100, d["wilson_hi"] * 100
        # From the counts: ``rate`` is stored to four places (0.4505 for
        # 50/111), and rounding that again gives 45.1% where 50/111 is 45.0%.
        rate = d["apps"] / p["denominator"] * 100
        c = ACCENT if headline else INK
        ax.plot([lo, hi], [i, i], color=c, linewidth=1.4, solid_capstyle="butt")
        for x in (lo, hi):
            ax.plot([x, x], [i - 0.13, i + 0.13], color=c, linewidth=1.4)
        ax.plot([rate], [i], "o", color=c, markersize=6, zorder=3)
        ax.text(hi + 2.0, i, f"{rate:.1f}%  [{lo:.1f}, {hi:.1f}]",
                va="center", fontsize=7.5, color=c)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 88)
    ax.set_xlabel("Percentage of LLM-integrated applications (n = 111), Wilson 95% CI")
    ax.grid(axis="x", color=LIGHT, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig6-2-prevalence-intervals")


def fig_liveness() -> None:
    """Figure 6.3 - credential disposition, showing the abstention explicitly."""
    lv = STATS["liveness"]
    provs = ["openai", "huggingface", "replicate", "deepseek"]
    names = {"openai": "OpenAI", "huggingface": "HuggingFace",
             "replicate": "Replicate", "deepseek": "DeepSeek"}
    live = [lv["by_provider"][p]["live"] for p in provs]
    revoked = [lv["by_provider"][p]["revoked"] for p in provs]
    unknown = [lv["by_provider"][p]["unknown"] for p in provs]

    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    y = range(len(provs))
    ax.barh(list(y), live, color=ACCENT, edgecolor=INK, linewidth=0.6,
            height=0.6, label="Live")
    ax.barh(list(y), revoked, left=live, color=LIGHT, edgecolor=INK,
            linewidth=0.6, height=0.6, label="Revoked")
    ax.barh(list(y), unknown, left=[l + r for l, r in zip(live, revoked)],
            color="white", edgecolor=INK, linewidth=0.6, height=0.6,
            hatch="///", label="Unverifiable (abstained)")
    ax.set_yticks(list(y))
    ax.set_yticklabels([names[p] for p in provs])
    ax.invert_yaxis()
    ax.set_xlabel("Credentials recovered")
    ax.set_xlim(0, 20)
    ax.set_xticks(range(0, 21, 2))   # credentials are integers
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(axis="x", color=LIGHT, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Live rate {lv['live_rate']*100:.0f}% "
                 f"({lv['live']} of {lv['live']+lv['revoked']} checked); "
                 f"{lv['unknown']} abstained",
                 fontsize=8.5, color=MID, loc="left", pad=8)
    save(fig, "fig6-3-credential-liveness")


def fig_sensitivity() -> None:
    """Figure 6.4 - how far the headline moves if the confirmation rule changes."""
    pairs = [
        ("Integration rate\n(of 1,495 scanned)", 7.42, 6.2, 8.9, 8.96, 7.6, 10.5),
        ("Inference-provider\ncredential", 35.1, 26.9, 44.4, 46.3, 38.0, 54.7),
        ("Any RQ1 artefact\n(headline)", 45.0, 36.1, 54.3, 54.5, 46.0, 62.7),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, (label, a, alo, ahi, b, blo, bhi) in enumerate(pairs):
        ya, yb = i + 0.16, i - 0.16
        ax.plot([alo, ahi], [ya, ya], color=INK, linewidth=1.4)
        ax.plot([a], [ya], "o", color=INK, markersize=5.5, zorder=3)
        ax.plot([blo, bhi], [yb, yb], color=ACCENT, linewidth=1.4, linestyle="--")
        ax.plot([b], [yb], "s", color=ACCENT, markersize=5.5, zorder=3)
        ax.text(ahi + 1.5, ya, f"{a:.1f}%", va="center", fontsize=7.5, color=INK)
        ax.text(bhi + 1.5, yb, f"{b:.1f}%", va="center", fontsize=7.5, color=ACCENT)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([p[0] for p in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0, 76)
    ax.set_xlabel("Percentage, Wilson 95% CI")
    ax.plot([], [], "o-", color=INK, label="As specified (§4.3), n = 111")
    ax.plot([], [], "s--", color=ACCENT, label="Credential accepted as confirming, n = 134")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left",
              bbox_to_anchor=(0.0, 1.01), ncol=2, handlelength=2.4,
              columnspacing=1.6, borderaxespad=0.0)
    ax.grid(axis="x", color=LIGHT, linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    save(fig, "fig6-4-confirmation-rule-sensitivity")


def fig_pipeline() -> None:
    """Figure 5.1 - the streaming pipeline, drawn to show the bounded staging
    area, which is the one property that makes corpus size independent of disk."""
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(6, 38)
    ax.axis("off")

    def box(x, y, w, h, text, fill="white", fs=7.5, lw=0.9):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK,
                               linewidth=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, zorder=3, linespacing=1.4)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, color=INK, linewidth=0.9,
                                     zorder=1))

    box(0, 16, 19, 10, "Frozen AndroZoo index\n27,589,444 rows", fs=7)
    arrow(19, 21, 23, 21)
    box(23, 16, 14, 10, "Frame +\nstratified\nsample")
    arrow(37, 21, 40, 21)

    # bounded staging area
    ax.add_patch(Rectangle((40, 8), 34, 26, facecolor="#f4f4f4",
                           edgecolor=MID, linewidth=0.8, linestyle="--", zorder=0))
    ax.text(57, 35.5, "bounded staging area  (16 slots, peak disk ≈ 272 MB)",
            ha="center", fontsize=7, color=MID, style="italic")
    box(41.5, 22, 15, 8, "download\n(thread pool)", fs=7)
    box(58.5, 22, 15, 8, "scan\n(process pool)", fs=7)
    box(50, 10, 14, 8, "delete /\nretain", fill="#ececec", fs=7)
    arrow(56.5, 26, 58.5, 26)
    arrow(66, 22, 60, 18)
    arrow(52, 18, 48, 22)

    arrow(74, 21, 79, 21)
    box(79, 16, 20, 10, "ledger (SQLite)\n+ detail store\n+ findings log")
    ax.text(89, 13.2, "resumable; keyed by SHA-256", ha="center",
            fontsize=6.8, color=MID, style="italic")
    save(fig, "fig5-1-streaming-pipeline")


if __name__ == "__main__":
    print("Building figures into dissertation/figures/")
    fig_pipeline()
    fig_funnel()
    fig_prevalence()
    fig_liveness()
    fig_sensitivity()
    print("done")
