#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot UCI regression outlier benchmark results.")
    parser.add_argument("--input", type=Path, default=Path("results/uci-regression-outliers.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input.open("rb") as f:
        data = pickle.load(f)

    if "preds" in data and "y" in data:
        y = data["y"]
        pred_map = data["preds"]
        time_map = data["time"]
    else:
        y = data["datasets"]["y"]
        pred_map = data["posterior-states"]
        time_map = data["time"]

    rmse = {k: np.sqrt(np.mean((pred - y) ** 2, axis=1)) for k, pred in pred_map.items()}

    rmse_df = pd.DataFrame(rmse).melt(var_name="method", value_name="rmse")
    plt.figure(figsize=(7, 3))
    sns.boxenplot(data=rmse_df, x="method", y="rmse")
    plt.xticks(rotation=30)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out1 = args.output_dir / "uci-regression-outliers-rmse.png"
    plt.savefig(out1, dpi=220)
    plt.close()

    time_df = pd.DataFrame(time_map).melt(var_name="method", value_name="seconds")
    plt.figure(figsize=(7, 3))
    sns.boxenplot(data=time_df, x="method", y="seconds")
    plt.xticks(rotation=30)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out2 = args.output_dir / "uci-regression-outliers-time.png"
    plt.savefig(out2, dpi=220)
    plt.close()

    print(f"Saved {out1}")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
