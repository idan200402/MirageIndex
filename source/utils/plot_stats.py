import argparse
import json
from html import escape
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT_PATH = DEFAULT_ARTIFACTS_DIR / "model_stats_comparison.svg"


def load_model_stats(artifacts_dir: Path) -> list[dict[str, float | str]]:
    stats = []

    for model_dir in sorted(path for path in artifacts_dir.iterdir() if path.is_dir()):
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_name = payload.get("model_name", model_dir.name)
        metrics = payload.get("metrics", payload)
        f1 = get_f1_score(metrics)
        accuracy = metrics.get("accuracy")

        if f1 is None:
            print(f"Skipping {model_name}: missing precision/recall/f1 in {metrics_path}")
            continue
        if accuracy is None:
            print(f"Skipping {model_name}: missing accuracy in {metrics_path}")
            continue

        stats.append(
            {
                "model_name": str(model_name),
                "f1": f1,
                "accuracy": float(accuracy),
            }
        )

    return stats


def get_f1_score(metrics: dict[str, Any]) -> float | None:
    if metrics.get("f1") is not None:
        return float(metrics["f1"])

    precision = metrics.get("precision")
    recall = metrics.get("recall")
    if precision is None or recall is None:
        return None

    precision = float(precision)
    recall = float(recall)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def build_svg(stats: list[dict[str, float | str]]) -> str:
    width = 980
    height = max(420, 170 + len(stats) * 96)
    margin_left = 190
    margin_right = 50
    margin_top = 90
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    group_height = plot_height / max(len(stats), 1)
    bar_height = min(28, max(18, group_height * 0.28))
    bar_gap = 8

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="36" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" font-weight="700">Model Stats Comparison</text>',
        f'<rect x="{margin_left}" y="54" width="14" height="14" fill="#2563eb" rx="2"/>',
        f'<text x="{margin_left + 22}" y="66" font-family="Arial, sans-serif" font-size="13" fill="#222">F1</text>',
        f'<rect x="{margin_left + 72}" y="54" width="14" height="14" fill="#16a34a" rx="2"/>',
        f'<text x="{margin_left + 94}" y="66" font-family="Arial, sans-serif" font-size="13" fill="#222">Accuracy</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1"/>',
    ]

    for tick in range(0, 11):
        value = tick / 10
        x = margin_left + value * plot_width
        lines.extend(
            [
                f'<line x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" stroke="#eeeeee" stroke-width="1"/>',
                f'<text x="{x:.1f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#444">{value:.1f}</text>',
            ]
        )

    for index, model_stats in enumerate(stats):
        model_name = str(model_stats["model_name"])
        f1_score = float(model_stats["f1"])
        accuracy = float(model_stats["accuracy"])
        group_y = margin_top + index * group_height
        f1_y = group_y + group_height / 2 - bar_height - bar_gap / 2
        accuracy_y = group_y + group_height / 2 + bar_gap / 2
        label_y = group_y + group_height / 2 + 5
        f1_width = f1_score * plot_width
        accuracy_width = accuracy * plot_width

        lines.extend(
            [
                f'<text x="{margin_left - 14}" y="{label_y:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="14" fill="#222">{escape(model_name)}</text>',
                f'<rect x="{margin_left}" y="{f1_y:.1f}" width="{f1_width:.1f}" height="{bar_height:.1f}" fill="#2563eb" rx="4"/>',
                f'<text x="{margin_left + f1_width + 8:.1f}" y="{f1_y + bar_height / 2 + 5:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#222">{f1_score:.4f}</text>',
                f'<rect x="{margin_left}" y="{accuracy_y:.1f}" width="{accuracy_width:.1f}" height="{bar_height:.1f}" fill="#16a34a" rx="4"/>',
                f'<text x="{margin_left + accuracy_width + 8:.1f}" y="{accuracy_y + bar_height / 2 + 5:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#222">{accuracy:.4f}</text>',
            ]
        )

    lines.append("</svg>")
    return "\n".join(lines)


def write_stats_plot(stats: list[dict[str, float | str]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_svg(stats), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an SVG plot comparing model F1 and accuracy scores.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory containing model artifact folders. Defaults to {DEFAULT_ARTIFACTS_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output SVG path. Defaults to {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = load_model_stats(args.artifacts_dir)

    if not stats:
        raise ValueError(f"No model stats found under {args.artifacts_dir}")

    output_path = write_stats_plot(stats, args.output)

    print("Model Stats")
    print("-----------")
    for model_stats in stats:
        print(
            f"{model_stats['model_name']}: "
            f"f1={float(model_stats['f1']):.4f}, "
            f"accuracy={float(model_stats['accuracy']):.4f}"
        )
    print(f"\nplot_path: {output_path}")


if __name__ == "__main__":
    main()
