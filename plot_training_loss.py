"""Plot training loss curves from Hugging Face TrainerState logs.

Reads `trainer_state.json` from a training checkpoint directory and writes
`loss_plot.png` so evidence includes both reward and loss trajectories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_log_history(checkpoint_dir: Path) -> List[Dict[str, Any]]:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"No trainer_state.json under {checkpoint_dir.resolve()!s}. "
            "Train first with train.py (or pass --checkpoint-dir to a run output dir)."
        )
    data = _load_json(state_path)
    history = data.get("log_history")
    if not history:
        raise ValueError(f"{state_path} has no log_history entries.")
    return history


def _loss_columns(columns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in columns:
        lc = c.lower()
        if lc == "loss":
            out.append(c)
        elif "loss" in lc:
            out.append(c)
    # Prioritize the canonical "loss", then keep deterministic order.
    return sorted(out, key=lambda x: (0 if x.lower() == "loss" else 1, x.lower()))


def _metric_columns(columns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for c in columns:
        lc = c.lower()
        if "loss" in lc or "reward" in lc:
            out.append(c)
    # Keep deterministic order and prioritize canonical keys first.
    priority = {"loss": 0, "reward": 1}
    return sorted(out, key=lambda x: (priority.get(x.lower(), 2), x.lower()))


def plot_loss_curves(log_history: List[Dict[str, Any]], out_path: Path, smooth: int = 0) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(log_history)
    if "step" not in df.columns:
        raise ValueError("log_history has no 'step' column.")

    loss_cols = _loss_columns([c for c in df.columns if c != "step"])
    if not loss_cols:
        raise ValueError(
            "No loss-like columns found in trainer_state log_history. "
            "Expected keys containing 'loss' (e.g., 'loss')."
        )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    steps = df["step"].to_numpy()
    colors = plt.cm.Set2.colors
    for i, col in enumerate(loss_cols):
        series = df[col].astype(float)
        y = series.rolling(window=smooth, min_periods=1).mean() if smooth > 1 else series
        label = f"{col} (rolling-{smooth})" if smooth > 1 else col
        ax.plot(steps, y.to_numpy(), label=label, color=colors[i % len(colors)], linewidth=2)

    ax.set_title("CogniMarket — Training loss progress")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_curves_separately(
    log_history: List[Dict[str, Any]],
    out_dir: Path,
    smooth: int = 0,
    show: bool = False,
) -> List[Path]:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(log_history)
    if "step" not in df.columns:
        raise ValueError("log_history has no 'step' column.")

    metric_cols = _metric_columns([c for c in df.columns if c != "step"])
    if not metric_cols:
        raise ValueError(
            "No loss/reward-like columns found in trainer_state log_history. "
            "Expected keys containing 'loss' or 'reward'."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    steps = df["step"].to_numpy()
    colors = plt.cm.Set2.colors
    written: List[Path] = []

    for i, col in enumerate(metric_cols):
        series = pd.to_numeric(df[col], errors="coerce")
        y = series.rolling(window=smooth, min_periods=1).mean() if smooth > 1 else series
        label = f"{col} (rolling-{smooth})" if smooth > 1 else col

        fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        ax.plot(steps, y.to_numpy(), label=label, color=colors[i % len(colors)], linewidth=2)
        ax.set_title(f"CogniMarket — {col} progress")
        ax.set_xlabel("Training step")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        out_path = out_dir / f"{col}_plot.png"
        fig.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        written.append(out_path)

    return written


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training loss from trainer_state.json")
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Directory containing trainer_state.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: <checkpoint-dir>/loss_plot.png)",
    )
    p.add_argument(
        "--separate",
        action="store_true",
        help="Write one plot per metric (loss/reward columns) instead of a single combined plot.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show generated figures (useful in notebooks).",
    )
    p.add_argument("--smooth", type=int, default=0, help="Rolling mean window (0 = off)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    history = load_log_history(args.checkpoint_dir)
    if args.separate:
        out_dir = args.out if args.out else (args.checkpoint_dir / "plots")
        written = plot_metric_curves_separately(
            history,
            out_dir=out_dir,
            smooth=args.smooth,
            show=args.show,
        )
        print(f"Wrote {len(written)} plot(s) to {out_dir.resolve()}")
        for path in written:
            print(f" - {path.resolve()}")
    else:
        out_path = args.out or (args.checkpoint_dir / "loss_plot.png")
        plot_loss_curves(history, out_path, smooth=args.smooth)
        print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()

