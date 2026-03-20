#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot online MLP training experiment outputs.")
    parser.add_argument("--input", type=Path, default=Path("results/online-mlp-training.pkl"))
    parser.add_argument("--output", type=Path, default=Path("figures/online-mlp-training-comparison.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("rb") as f:
        data = pickle.load(f)

    methods = list(data["results"].keys())
    nrows = 2
    ncols = int(np.ceil(len(methods) / nrows))

    fig, axs = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), sharex=True, sharey=True)
    axs = np.array(axs).reshape(-1)

    for ax, method in zip(axs, methods):
        ax.scatter(data["x"], data["results"][method]["one_step"], s=8, label=method)
        ax.plot(data["xtest"], data["ytest"], color="black", linewidth=1.5)
        ax.grid(alpha=0.3)
        ax.set_title(method)

    for ax in axs[len(methods):]:
        ax.axis("off")

    fig.supxlabel("x")
    fig.supylabel("y")
    plt.tight_layout()
    plt.savefig(args.output, dpi=220)
    plt.close(fig)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
