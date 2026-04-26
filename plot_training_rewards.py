"""Plot GRPO training reward curves from a finished Hugging Face Trainer run.

TRL's GRPOTrainer logs (among others):

- ``reward`` / ``reward_std`` — mean and std of the (weighted) sum of reward heads
- ``rewards/<callable.__name__>/mean`` — per reward function (e.g. outcome_reward_fn)

This script reads ``trainer_state.json`` from your checkpoint directory (same folder
where ``train.py`` sets ``--save-dir``) and writes a PNG suitable for rubrics that
ask for evidence of learning (curves + optional before/after eval bars).

Examples
--------
After training (default save dir)::

    python plot_training_rewards.py

Custom paths::

    python plot_training_rewards.py --checkpoint-dir ./my_run --out ./figures/rewards.png

If ``training_progress.json`` exists next to ``trainer_state.json`` (written by
``train.py`` after a full run), a second panel compares **rule baseline** (equal
split), **pre-train model**, and **post-train model** on the same env seeds when
``baseline`` is present in the JSON. Older JSON files with only ``pre``/``post``
still plot two bars.

To add a baseline to an old ``training_progress.json`` that lacks ``baseline``,
re-run this script with ``--baseline-json`` pointing at a small JSON object
containing ``avg_reward``, ``success_rate``, ``avg_utility``, and optionally
``avg_rounds`` / ``policy`` (see ``evaluate.run_rule_baseline_metrics``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_log_history(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"No trainer_state.json under {checkpoint_dir.resolve()!s}. "
            "Train first with train.py (or point --checkpoint-dir at your GRPO output_dir)."
        )
    data = _load_json(state_path)
    history = data.get("log_history")
    if not history:
        raise ValueError(f"{state_path} has no log_history — training may not have logged yet.")
    return history


def _reward_columns(columns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in columns:
        if c in ("reward", "reward_std"):
            out.append(c)
        elif c.startswith("rewards/") and c.endswith("/mean"):
            out.append(c)
        elif c.startswith("reward/") and len(c) > 7 and c.split("/")[1].isdigit():
            # Older / alternate logging style
            out.append(c)
    # Stable order: total first, then alphabetical per-head
    priority = {"reward": 0, "reward_std": 1}
    return sorted(out, key=lambda x: (priority.get(x, 10), x))


def _bar_values(row: Dict[str, Any]) -> List[float]:
    return [float(row["avg_reward"]), float(row["success_rate"]) * 100, float(row["avg_utility"])]


def plot_reward_curves(
    log_history: List[Dict[str, Any]],
    out_path: Path,
    *,
    title: str = "CogniMarket — GRPO reward progress",
    smooth: int = 0,
    pre_post_eval: Optional[Dict[str, Any]] = None,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(log_history)
    if "step" not in df.columns:
        raise ValueError("log_history has no 'step' column — unexpected TrainerState format.")

    metric_cols = _reward_columns([c for c in df.columns if c != "step"])
    if not metric_cols:
        raise ValueError(
            "No reward columns found in log_history. "
            "Expected keys like 'reward', 'reward_std', or 'rewards/*/mean'."
        )

    has_bar_panel = (
        pre_post_eval is not None
        and "pre" in pre_post_eval
        and "post" in pre_post_eval
    )
    fig, axes = plt.subplots(
        2 if has_bar_panel else 1,
        1,
        figsize=(10, 7 if has_bar_panel else 5),
        height_ratios=[2.2, 1] if has_bar_panel else [1],
        constrained_layout=True,
    )
    ax_curve = axes[0] if has_bar_panel else axes

    steps = df["step"].to_numpy()
    colors = plt.cm.tab10.colors
    for i, col in enumerate(metric_cols):
        series = df[col].astype(float)
        if smooth > 1:
            y = series.rolling(window=smooth, min_periods=1).mean()
            label = f"{col} (rolling-{smooth})"
        else:
            y = series
            label = col
        ax_curve.plot(
            steps,
            y.to_numpy(),
            label=label,
            color=colors[i % len(colors)],
            linewidth=2 if col == "reward" else 1.2,
        )

    ax_curve.set_title(title)
    ax_curve.set_xlabel("Training step")
    ax_curve.set_ylabel("Reward (logged)")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="best", fontsize=8)

    if has_bar_panel:
        ax_bar = axes[1]
        pre = pre_post_eval["pre"]
        post = pre_post_eval["post"]
        baseline = pre_post_eval.get("baseline")
        labels_txt = ["Avg reward", "Success %", "Avg utility"]
        x = np.arange(len(labels_txt))
        b_pre = _bar_values(pre)
        b_post = _bar_values(post)

        if baseline and all(k in baseline for k in ("avg_reward", "success_rate", "avg_utility")):
            b_base = _bar_values(baseline)
            w = 0.22
            blabel = str(baseline.get("policy", "rule baseline")).replace("_", " ")
            ax_bar.bar(x - w, b_base, width=w, label=blabel, color="#7f7f7f", alpha=0.9)
            ax_bar.bar(x, b_pre, width=w, label="Pre-training model", color="#4c72b0", alpha=0.9)
            ax_bar.bar(x + w, b_post, width=w, label="Post-training model", color="#55a868", alpha=0.9)
        else:
            w = 0.35
            ax_bar.bar(x - w / 2, b_pre, width=w, label="Pre-training model", alpha=0.85)
            ax_bar.bar(x + w / 2, b_post, width=w, label="Post-training model", alpha=0.85)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels_txt)
        ax_bar.set_ylabel("Value")
        ax_bar.legend(loc="best", fontsize=8)
        ax_bar.grid(True, axis="y", alpha=0.3)
        ax_bar.set_title("Environment evaluation (same seeds across series)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reward_curves_separately(
    log_history: List[Dict[str, Any]],
    out_dir: Path,
    *,
    smooth: int = 0,
    show: bool = False,
) -> List[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(log_history)
    if "step" not in df.columns:
        raise ValueError("log_history has no 'step' column — unexpected TrainerState format.")

    metric_cols = _reward_columns([c for c in df.columns if c != "step"])
    if not metric_cols:
        raise ValueError(
            "No reward columns found in log_history. "
            "Expected keys like 'reward', 'reward_std', or 'rewards/*/mean'."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    steps = df["step"].to_numpy()
    colors = plt.cm.tab10.colors
    written: List[Path] = []

    for i, col in enumerate(metric_cols):
        series = pd.to_numeric(df[col], errors="coerce")
        y = series.rolling(window=smooth, min_periods=1).mean() if smooth > 1 else series
        label = f"{col} (rolling-{smooth})" if smooth > 1 else col

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        ax.plot(
            steps,
            y.to_numpy(),
            label=label,
            color=colors[i % len(colors)],
            linewidth=2 if col == "reward" else 1.2,
        )
        ax.set_title(f"CogniMarket — {col} progress")
        ax.set_xlabel("Training step")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        safe_name = col.replace("/", "_")
        out_path = out_dir / f"{safe_name}_plot.png"
        fig.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        written.append(out_path)

    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot GRPO reward curves from trainer_state.json")
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Directory containing trainer_state.json (train.py --save-dir).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <checkpoint-dir>/reward_plot.png).",
    )
    p.add_argument(
        "--separate",
        action="store_true",
        help="Write one reward plot per metric into --out directory.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show generated figures (useful in notebooks).",
    )
    p.add_argument("--smooth", type=int, default=0, help="Rolling-mean window (0 = off).")
    p.add_argument(
        "--progress-json",
        type=Path,
        default=None,
        help="Optional training_progress.json for pre/post eval bars "
        "(default: <checkpoint-dir>/training_progress.json if present).",
    )
    p.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional JSON with keys avg_reward, success_rate, avg_utility (and optional policy). "
        "Merged into the eval bar panel if training_progress.json has no 'baseline' key.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    ckpt: Path = args.checkpoint_dir

    history = load_log_history(ckpt)
    pre_post: Optional[Dict[str, Dict[str, float]]] = None
    progress_path = args.progress_json
    if progress_path is None:
        candidate = ckpt / "training_progress.json"
        if candidate.is_file():
            progress_path = candidate
    if progress_path is not None and progress_path.is_file():
        raw = _load_json(progress_path)
        if isinstance(raw, dict) and "pre" in raw and "post" in raw:
            pre_post = {"pre": raw["pre"], "post": raw["post"]}
            if isinstance(raw.get("baseline"), dict):
                pre_post["baseline"] = raw["baseline"]
    if pre_post is not None and args.baseline_json is not None and args.baseline_json.is_file():
        bl = _load_json(args.baseline_json)
        if isinstance(bl, dict) and "baseline" not in pre_post:
            pre_post["baseline"] = bl

    if args.separate:
        out_dir: Path = args.out if args.out else (ckpt / "reward_plots")
        written = plot_reward_curves_separately(
            history,
            out_dir,
            smooth=args.smooth,
            show=args.show,
        )
        print(f"Wrote {len(written)} plot(s) to {out_dir.resolve()}")
        for path in written:
            print(f" - {path.resolve()}")
    else:
        out_path: Path = args.out or (ckpt / "reward_plot.png")
        plot_reward_curves(
            history,
            out_path,
            title="CogniMarket — GRPO training reward progress",
            smooth=args.smooth,
            pre_post_eval=pre_post,
        )
        print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
