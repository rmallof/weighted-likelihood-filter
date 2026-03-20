#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gamma-process filtering results.")
    parser.add_argument("--input", type=Path, default=Path("results/gamma-process.pkl"))
    parser.add_argument("--output", type=Path, default=Path("figures/gamma-process-comparison.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("rb") as f:
        data = pickle.load(f)

    plt.figure(figsize=(8, 4))
    plt.plot(data["pred"]["KF"], label="KF")
    plt.plot(data["pred"]["KF-IW"], label="KF-IW")
    plt.plot(data["pred"]["WLF-IMQ"], label="WLF-IMQ")
    plt.plot(data["state"], label="state", color="black", linewidth=1.5)
    plt.grid(alpha=0.3)
    plt.xlabel("t")
    plt.ylabel("y_t")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    plt.close()

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
