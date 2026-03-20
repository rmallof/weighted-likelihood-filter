#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


C_MAP = {
    "KF-IW": "crimson",
    "WoLF-IMQ": "dodgerblue",
    "KF": "lightseagreen",
    "WoLF-MD": "gold",
    "KF-B": "darkorange",
}

TITLES = {
    "mean": "Mixture",
    "covariance": "Student",
}


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot 2D tracking results from saved pickle files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=base_dir / "results",
        help="Directory with 2d-ssm-outlier-*.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "figures",
        help="Directory to save figure files.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["mean", "covariance"],
        choices=["mean", "covariance", "added_mean"],
        help="Result kinds to plot.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=1,
        help="Trajectory index for the single-run comparison figure.",
    )
    return parser.parse_args()


def load_results(input_dir: Path, kinds: list[str]) -> dict:
    data = {}
    for kind in kinds:
        path = input_dir / f"2d-ssm-outlier-{kind}.pkl"
        with path.open("rb") as f:
            data[kind] = pickle.load(f)
    return data


def build_diff_df(data: dict) -> pd.DataFrame:
    statev = data["datasets"]["latent"]
    err_methods = jax.tree.map(lambda x: x - statev, data["posterior-states"])

    diff_df = jax.tree.map(lambda x: np.sum(x**2, axis=1), err_methods)
    diff_df = pd.concat(
        [pd.DataFrame(diff_df[k]).reset_index().melt("index").assign(method=k) for k in diff_df]
    )
    diff_df = diff_df.rename({"variable": "state", "value": "error", "index": "trial"}, axis=1)
    diff_df["kind"] = data["name"]
    return diff_df


def plot_per_kind_error(diff_df: pd.DataFrame, output_file: Path) -> None:
    methods = sorted(diff_df.method.unique())
    ax = sns.boxenplot(
        y="error",
        x="state",
        hue="method",
        data=diff_df,
        palette=C_MAP,
        hue_order=methods,
    )
    legend = ax.legend(ncol=2)
    if legend is not None:
        for text in legend.get_texts():
            text.set_text(text.get_text().replace("-MD", "-TMD"))
    plt.xlabel("State component")
    plt.ylabel("$J_{T,i}$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(output_file, dpi=300)
    plt.close()


def plot_single_run(data_single: dict, kind: str, sample_index: int, output_file: Path) -> None:
    latent_i = data_single["datasets"]["latent"][sample_index, :, :2].T
    observed_i = data_single["datasets"]["observed"][sample_index]
    hist_runs = jax.tree.map(lambda x: x[sample_index, :, :2], data_single["posterior-states"])

    fig, axs = plt.subplots(3, 2, figsize=(6.4 * 0.5, 4.2))
    fs = 15

    model_axes = list(axs.ravel()[1:])
    for ax_i, key in zip(model_axes, hist_runs):
        state_est = hist_runs[key]
        ax_i.plot(*state_est.T, c=C_MAP[key], linewidth=2)
        ax_i.set_title(key.replace("-MD", "-TMD"), fontsize=fs)
        ax_i.axis("off")
        ax_i.scatter(0, 0, c="black", zorder=3)

    ref_ax = model_axes[0]
    ylim = ref_ax.get_ylim()
    xlim = ref_ax.get_xlim()

    axs[0, 0].scatter(*observed_i.T, c="none", edgecolor="tab:gray", alpha=0.5, s=10)
    axs[0, 0].plot(*latent_i, c="black", linewidth=2.0)
    axs[0, 0].axis("off")
    axs[0, 0].scatter(0, 0, c="black", zorder=3)
    axs[0, 0].set_ylim(*ylim)
    axs[0, 0].set_xlim(*xlim)
    axs[0, 0].set_title("truth", fontsize=fs)

    plt.suptitle(TITLES.get(kind, kind), y=0.9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_combined_error(data: dict, output_file: Path) -> None:
    if not {"mean", "covariance"}.issubset(data.keys()):
        return

    diff_df = pd.concat([build_diff_df(data["mean"]), build_diff_df(data["covariance"])], axis=0)
    diff_df["kind"] = diff_df["kind"].apply(lambda x: TITLES.get(x, x))

    methods = sorted(diff_df.method.unique())
    ax = sns.boxenplot(
        y="error",
        x="kind",
        hue="method",
        data=diff_df.query("state == 0"),
        palette=C_MAP,
        hue_order=methods,
        order=["Student", "Mixture"],
    )
    legend = ax.legend(ncol=2)
    if legend is not None:
        for text in legend.get_texts():
            text.set_text(text.get_text().replace("-MD", "-TMD"))

    plt.xlabel("Variant")
    plt.ylabel("$J_{T,0}$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(output_file, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_results(args.input_dir, args.kinds)

    for kind, data_single in data.items():
        diff_df = build_diff_df(data_single)
        plot_per_kind_error(
            diff_df,
            args.output_dir / f"2d-ssm-comparison-outlier-{kind}.png",
        )
        plot_single_run(
            data_single,
            kind,
            args.sample_index,
            args.output_dir / f"2d-ssm-comparison-single-run-{kind}.png",
        )

        time_df = pd.DataFrame(data_single["time"]).iloc[1:]
        speed = ((time_df / time_df["KF"].median(axis=0)).describe().drop("KF", axis=1)).T
        print(f"{kind} median runtime relative to KF:")
        print(speed["50%"].round(1))

    plot_combined_error(data, args.output_dir / "2d-ssm-comparison-outlier-both.png")


if __name__ == "__main__":
    main()
