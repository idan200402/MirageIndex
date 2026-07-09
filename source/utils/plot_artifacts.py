"""Unified visualization tool for model training/evaluation artifacts.

Every model writes a ``metrics.json`` file under ``artifacts/<model_name>/`` (see
``training_metrics.export_metrics_json``). This tool reads those files and renders
several complementary views with matplotlib, each selectable from the command line.

The four views answer different questions:

* ``metrics``         -- how do all models compare across every test metric?
* ``spans-compare``   -- what did the "spans" variant change vs. its regular twin?
* ``training-curves`` -- how did train/val loss evolve for the neural models?
* ``span-coverage``   -- how much positive signal did span resolution recover?

Example
-------
    python -m source.utils.plot_artifacts --plot spans-compare --metric pr_auc
    python -m source.utils.plot_artifacts --plot metrics --output artifacts/all_metrics.png
    python -m source.utils.plot_artifacts --plot training-curves --show
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # safe default: render to file without a display server
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Suffix that distinguishes a "spans" artifact from its regular twin.
SPANS_SUFFIX = "_spans"

# The five metrics every metrics.json exposes under the "metrics" key.
METRIC_KEYS = ["accuracy", "precision", "recall", "pr_auc", "roc_auc"]

# Colorblind-safe categorical palette (fixed order, never cycled). Slots are
# assigned to *entities* so a filtered-out series never repaints its neighbours.
PALETTE = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}

# Roles pulled straight from the reference palette so both variants read as a set.
REGULAR_COLOR = PALETTE["blue"]
SPANS_COLOR = PALETTE["orange"]
TRAIN_COLOR = PALETTE["blue"]
VAL_COLOR = PALETTE["orange"]

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_all_metrics(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    """Read every ``metrics.json`` under ``artifacts_dir`` into ``{name: payload}``.

    The dictionary key is the artifact folder name (e.g. ``tfidf_xgboost`` or
    ``tfidf_xgboost_spans``) so regular and spans variants stay distinct.
    """
    payloads: dict[str, dict[str, Any]] = {}
    for model_dir in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        payloads[model_dir.name] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payloads


def base_name(name: str) -> str:
    """Strip the spans suffix so a variant can be matched to its regular twin."""
    return name[: -len(SPANS_SUFFIX)] if name.endswith(SPANS_SUFFIX) else name


def paired_variants(
    payloads: dict[str, dict[str, Any]]
) -> list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]]:
    """Group payloads into ``(base, regular_payload, spans_payload)`` rows.

    A model that only exists in one variant still appears, with ``None`` for the
    missing side, so the caller can decide whether to show or skip it.
    """
    bases = sorted({base_name(name) for name in payloads})
    rows = []
    for base in bases:
        regular = payloads.get(base)
        spans = payloads.get(base + SPANS_SUFFIX)
        rows.append((base, regular, spans))
    return rows


def _metrics_of(payload: dict[str, Any]) -> dict[str, float]:
    """Return the inner metrics dict, tolerating a flat legacy layout."""
    return payload.get("metrics", payload)


# --------------------------------------------------------------------------- #
# Shared styling helpers
# --------------------------------------------------------------------------- #
def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


# --------------------------------------------------------------------------- #
# Plot: metrics -- every model across every metric
# --------------------------------------------------------------------------- #
def plot_metrics(payloads: dict[str, dict[str, Any]]) -> plt.Figure:
    """Small multiples: one panel per metric, all models as sorted bars.

    A grouped bar chart coloured by model would break past eight series (the
    categorical palette can no longer stay colourblind-safe). Faceting by metric
    and sorting each panel keeps the comparison legible for any model count and
    uses a single hue, so no colour is asked to carry identity it cannot.
    """
    names = sorted(payloads)
    n = len(METRIC_KEYS)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    height = max(3.4, 0.32 * len(names)) * rows
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, height), squeeze=False)

    for idx, metric in enumerate(METRIC_KEYS):
        ax = axes[idx // cols][idx % cols]
        scored = sorted(
            ((name, _metrics_of(payloads[name]).get(metric) or 0.0) for name in names),
            key=lambda item: item[1],
        )
        labels = [name for name, _ in scored]
        values = [value for _, value in scored]
        # Colour the spans variants distinctly so the two families read apart
        # without asking colour to encode 15 separate identities.
        colors = [SPANS_COLOR if name.endswith(SPANS_SUFFIX) else REGULAR_COLOR
                  for name in labels]
        y = range(len(labels))
        ax.barh(list(y), values, color=colors)
        for i, value in enumerate(values):
            ax.text(value + 0.01, i, f"{value:.3f}", va="center",
                    fontsize=7, color=MUTED)

        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_title(metric.replace("_", " ").upper(), color=INK, fontsize=11)
        _style_axis(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)

    # Legend (regular vs spans) once for the whole figure; blank unused panels.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=REGULAR_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=SPANS_COLOR),
    ]
    axes[0][cols - 1].legend(handles, ["regular", "spans"], fontsize=8,
                             frameon=False, loc="lower right")
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("Model metrics across the held-out test set (sorted per metric)",
                 color=INK, fontsize=14, y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Plot: spans-compare -- regular vs spans, side by side (REQUIRED view)
# --------------------------------------------------------------------------- #
def plot_spans_compare(
    payloads: dict[str, dict[str, Any]], ax: plt.Axes, metric: str
) -> None:
    """Paired bars: the regular vs spans value of one metric for each model.

    Only models that have *both* variants are shown, because the whole point is
    the head-to-head. A signed delta label (spans − regular) makes the direction
    and size of each change obvious.
    """
    if metric not in METRIC_KEYS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {METRIC_KEYS}")

    rows = [
        (base, reg, spn)
        for base, reg, spn in paired_variants(payloads)
        if reg is not None and spn is not None
    ]
    if not rows:
        raise ValueError("No model has both a regular and a spans variant to compare.")

    labels = [base for base, _, _ in rows]
    y = range(len(rows))
    bar_h = 0.36

    for i, (_, reg, spn) in enumerate(rows):
        reg_val = _metrics_of(reg).get(metric) or 0.0
        spn_val = _metrics_of(spn).get(metric) or 0.0

        ax.barh(i + bar_h / 2 + 0.02, reg_val, bar_h, color=REGULAR_COLOR,
                label="regular" if i == 0 else None)
        ax.barh(i - bar_h / 2 - 0.02, spn_val, bar_h, color=SPANS_COLOR,
                label="spans" if i == 0 else None)

        # Signed delta (spans - regular) annotated at the longer of the two bars.
        delta = spn_val - reg_val
        sign = "+" if delta >= 0 else "−"
        ax.text(max(reg_val, spn_val) + 0.01, i,
                f"{sign}{abs(delta):.3f}", va="center", fontsize=8, color=MUTED)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title(
        f"Regular vs spans — {metric.replace('_', ' ').upper()} "
        "(delta = spans − regular)",
        color=INK, fontsize=13,
    )
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)


# --------------------------------------------------------------------------- #
# Plot: training-curves -- train/val loss for neural models
# --------------------------------------------------------------------------- #
def plot_training_curves(payloads: dict[str, dict[str, Any]]) -> plt.Figure:
    """Small multiples of train/val loss vs. epoch for models that logged history.

    Each subplot is one model; the best epoch (lowest val loss the trainer kept)
    is marked. Only payloads carrying ``training_history`` qualify, so tf-idf and
    majority-vote baselines are silently skipped.
    """
    trainable = {
        name: payload
        for name, payload in sorted(payloads.items())
        if payload.get("training_history")
    }
    if not trainable:
        raise ValueError("No artifact contains a 'training_history' to plot.")

    n = len(trainable)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)

    for idx, (name, payload) in enumerate(trainable.items()):
        ax = axes[idx // cols][idx % cols]
        history = payload["training_history"]
        epochs = [entry["epoch"] for entry in history]
        train_loss = [entry.get("train_loss") for entry in history]
        val_loss = [entry.get("val_loss") for entry in history]

        # History may concatenate several learning-rate sweeps; plot against a
        # monotonic step index so the concatenation reads left-to-right.
        steps = range(1, len(history) + 1)
        ax.plot(steps, train_loss, color=TRAIN_COLOR, linewidth=2, label="train")
        ax.plot(steps, val_loss, color=VAL_COLOR, linewidth=2, label="val")

        if any(v is not None for v in val_loss):
            best_i = min(
                (i for i, v in enumerate(val_loss) if v is not None),
                key=lambda i: val_loss[i],
            )
            ax.scatter([best_i + 1], [val_loss[best_i]], color=VAL_COLOR,
                       s=45, zorder=5, edgecolor="white", linewidth=1.2)
            ax.annotate(f"best val {val_loss[best_i]:.3f}",
                        (best_i + 1, val_loss[best_i]),
                        textcoords="offset points", xytext=(6, 8),
                        fontsize=8, color=MUTED)

        ax.set_title(name, color=INK, fontsize=11)
        ax.set_xlabel("training step (epochs, LR sweeps concatenated)")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8, frameon=False)
        _style_axis(ax)
        ax.grid(axis="both", color=GRID, linewidth=0.8)

    # Blank any unused axes in the final row.
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("Neural training curves", color=INK, fontsize=14, y=1.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Plot: span-coverage -- how much positive signal spans recovered
# --------------------------------------------------------------------------- #
def plot_span_coverage(payloads: dict[str, dict[str, Any]], ax: plt.Axes) -> None:
    """Stacked bars of resolved vs zero-span 'yes' records per spans model.

    ``span_coverage`` records how many positive examples had at least one span
    resolved (``yes_records_resolved``) versus none (``yes_records_zero_spans``).
    A high zero-span share means the span pipeline discarded real positives.
    """
    rows = [
        (name, payload["span_coverage"])
        for name, payload in sorted(payloads.items())
        if payload.get("span_coverage")
    ]
    if not rows:
        raise ValueError("No artifact contains 'span_coverage' (spans models only).")

    labels = [name for name, _ in rows]
    resolved = [cov.get("yes_records_resolved", 0) for _, cov in rows]
    zero = [cov.get("yes_records_zero_spans", 0) for _, cov in rows]
    y = range(len(rows))

    ax.barh(list(y), resolved, color=PALETTE["aqua"], label="resolved (≥1 span)")
    ax.barh(list(y), zero, left=resolved, color=PALETTE["red"], label="zero spans")

    for i, (_, cov) in enumerate(rows):
        total = cov.get("yes_records_total") or (resolved[i] + zero[i])
        pct = 100 * resolved[i] / total if total else 0.0
        ax.text(resolved[i] + zero[i] + max(resolved + zero) * 0.01, i,
                f"{pct:.0f}% resolved", va="center", fontsize=8, color=MUTED)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("'yes' records")
    ax.set_title("Span coverage of positive records", color=INK, fontsize=13)
    # Bars are all near-equal length, so there is no empty interior; park the
    # legend just outside the top-right (bbox_inches="tight" keeps it in frame).
    ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
PLOT_CHOICES = ["metrics", "spans-compare", "training-curves", "span-coverage"]


def _default_output(plot: str, metric: str) -> Path:
    stem = plot if plot != "spans-compare" else f"spans_compare_{metric}"
    return DEFAULT_ARTIFACTS_DIR / f"{stem}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize model artifacts in several complementary ways.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--plot", choices=PLOT_CHOICES, required=True,
        help="Which visualization to render.",
    )
    parser.add_argument(
        "--metric", choices=METRIC_KEYS, default="pr_auc",
        help="Metric to compare (used by --plot spans-compare).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Restrict to these artifact folder names (default: all found).",
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing per-model artifact folders.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output image path (default: artifacts/<plot>.png).",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Raster resolution for PNG output.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open an interactive window instead of only writing the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    payloads = load_all_metrics(args.artifacts_dir)
    if not payloads:
        raise ValueError(f"No metrics.json files found under {args.artifacts_dir}")

    if args.models:
        missing = [m for m in args.models if m not in payloads]
        if missing:
            raise ValueError(f"Requested models not found: {missing}")
        payloads = {name: payloads[name] for name in args.models}

    if args.show:
        # Swap to an interactive backend only when explicitly requested.
        matplotlib.use("TkAgg", force=True)

    # metrics and training-curves build their own multi-panel figures; the
    # single-axis views share one Figure/Axes.
    if args.plot == "metrics":
        fig = plot_metrics(payloads)
    elif args.plot == "training-curves":
        fig = plot_training_curves(payloads)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        if args.plot == "spans-compare":
            plot_spans_compare(payloads, ax, args.metric)
        elif args.plot == "span-coverage":
            plot_span_coverage(payloads, ax)
        fig.tight_layout()

    output_path = args.output or _default_output(args.plot, args.metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"plot_path: {output_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
