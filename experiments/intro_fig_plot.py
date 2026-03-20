#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot intro-fig experiment outputs.")
    parser.add_argument("--input", type=Path, default=base_dir / "results" / "intro-fig.pkl")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input.open("rb") as f:
        data = pickle.load(f)

    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 3

    measurements = np.asarray(data["measurements"])
    state_arr = np.asarray(data["state"])
    hist_kf_mean = np.asarray(data["hist"]["KF"]["mean"])
    hist_kf_cov = np.asarray(data["hist"]["KF"]["cov"])
    hist_imq_mean = np.asarray(data["hist"]["WLF-IMQ"]["mean"])
    hist_imq_cov = np.asarray(data["hist"]["WLF-IMQ"]["cov"])
    xout = np.asarray(data["xout"]).astype(int)
    is_outlier = np.asarray(data["is_outlier"]).astype(bool)

    ix = 1
    y = measurements[:, ix]
    yout = y[xout]

    lw = 4
    ix = 0
    xrange = np.arange(len(measurements))
    y = measurements[:, ix]

    std_kf = np.sqrt(hist_kf_cov[:, ix, ix])
    std_wlkf = np.sqrt(hist_imq_cov[:, ix, ix])

    fig, _ = plt.subplots(figsize=(6.4 * 1.1, 4.8 * 0.9))
    c = "tab:blue"
    ubound = hist_kf_mean[:, ix] + 2 * std_kf
    lbound = hist_kf_mean[:, ix] - 2 * std_kf
    plt.fill_between(xrange, ubound, lbound, color=c, alpha=0.3)
    plt.plot(xrange, hist_kf_mean[:, ix], c=c, label="KF", linewidth=lw, linestyle="dotted")

    c = "tab:orange"
    ubound = hist_imq_mean[:, ix] + 2 * std_wlkf
    lbound = hist_imq_mean[:, ix] - 2 * std_wlkf
    plt.fill_between(xrange, ubound, lbound, color=c, alpha=0.3)
    plt.plot(xrange, hist_imq_mean[:, ix], c=c, label="WLF", linewidth=lw)

    plt.plot(xrange, state_arr[:, ix], c="black", linestyle="--", label="KF-clean", linewidth=lw)
    plt.plot(xrange[~is_outlier], y[~is_outlier], c="tab:gray", marker="o", linewidth=0, markersize=8)
    plt.scatter(xout, yout, c="tab:red", zorder=3, s=50, marker="x", linewidth=4)
    plt.xticks([])
    plt.yticks([])
    plt.ylabel("$y_t$")
    plt.xlabel("$t$")
    plt.tight_layout()
    out1 = args.output_dir / "intro-fig-trajectories.png"
    plt.savefig(out1, dpi=300)
    plt.close()

    klds = data["pif_grid"]
    domain = np.asarray(data["domain"])

    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(6.4 * 1.2, 4.8 * 0.8))
    norm = Normalize(vmin=0, vmax=16)
    cmap = "RdBu_r"
    c0 = ax[0].contourf(*domain, np.asarray(klds["IMQ"]), cmap=cmap, norm=norm)
    c1 = ax[1].contourf(*domain, np.asarray(klds["MD"]), cmap=cmap, norm=norm)
    c2 = ax[2].contourf(*domain, np.asarray(klds["KF"]), cmap=cmap, norm=norm)

    ax[1].set_xlabel(r"$\epsilon_1$", labelpad=-5)
    ax[0].set_ylabel(r"$\epsilon_2$", labelpad=-10)
    ax[0].set_title("WLF-IMQ")
    ax[1].set_title("WLF-TMD")
    ax[2].set_title("KF")
    plt.colorbar(c2, ax=ax, location="bottom", label="PIF", pad=0.2)

    out2 = args.output_dir / "intro-fig-pif-grid.png"
    plt.savefig(out2, dpi=300)
    plt.close(fig)

    print(f"Saved {out1}")
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
