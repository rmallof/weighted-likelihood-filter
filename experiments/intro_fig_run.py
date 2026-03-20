#!/usr/bin/env python3

import argparse
import pickle
from functools import partial
from pathlib import Path

import datagen
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from ssm_robust_filters import LinearSSMRobustFilters


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run intro-fig experiment and store trajectories + PIF grids.")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--output", type=Path, default=base_dir / "results" / "intro-fig.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    dgen = datagen.GaussMeanOutlierMovingObject2D(
        1.0,
        0.1,
        1.0,
        outlier_proba=0.0,
        outlier_scale=2.0,
    )

    initial_mean = jnp.array([0.0, 0.0, 1.0, 1.0])
    dataset = dgen.sample(key, initial_mean, args.n_steps)
    measurements_clean = dataset["observed"]
    state = dataset["latent"]

    xout = jnp.array([5, 10, 15])
    measurements_out = measurements_clean[xout] * 2 * jnp.array([-1, 1, -1])[:, None]
    measurements = measurements_clean.at[xout].set(measurements_out)
    tfd = tfp.distributions

    ssm_filters = LinearSSMRobustFilters(
        transition_matrix=dgen.transition_matrix,
        projection_matrix=dgen.projection_matrix,
        observation_covariance=dgen.observation_covariance,
        dynamics_covariance=dgen.dynamics_covariance,
    )

    bel0_m = initial_mean
    bel0_p = jnp.eye(initial_mean.shape[0])
    _, hist_kf = ssm_filters.scan_kf(measurements, bel0_m, bel0_p)
    _, hist_imq = ssm_filters.scan_imq(measurements, bel0_m, bel0_p, soft_threshold=2.0)

    bel_base, _ = ssm_filters.scan_kf(measurements_clean, bel0_m, bel0_p)
    norm_base = tfd.MultivariateNormalFullCovariance(loc=bel_base.mean, covariance_matrix=bel_base.cov)

    xout_last = -1

    @jax.jit
    def pif_last(errx, erry):
        verr = jnp.array([errx, erry])
        meas = measurements_clean.at[xout_last].set(measurements_clean[-1] + verr)

        bel_imq_, out = ssm_filters.scan_imq(meas, bel0_m, bel0_p, soft_threshold=4.0)
        w_imq = out["weight"][-1]
        mean_pred_imq = out["mean_pred"][-1]
        norm_imq = tfd.MultivariateNormalFullCovariance(
            loc=bel_imq_.mean,
            covariance_matrix=bel_imq_.cov,
        )

        bel_md, out = ssm_filters.scan_md(meas, bel0_m, bel0_p, threshold=4.0)
        w_md = out["weight"][-1]
        mean_pred_md = out["mean_pred"][-1]
        norm_md = tfd.MultivariateNormalFullCovariance(
            loc=bel_md.mean,
            covariance_matrix=bel_md.cov,
        )

        bel_kf_, out = ssm_filters.scan_kf(meas, bel0_m, bel0_p)
        w_kf = out["weight"][-1]
        mean_pred_kf = out["mean_pred"][-1]
        norm_kf = tfd.MultivariateNormalFullCovariance(
            loc=bel_kf_.mean,
            covariance_matrix=bel_kf_.cov,
        )

        return {
            "kld": {
                "IMQ": norm_base.kl_divergence(norm_imq),
                "MD": norm_base.kl_divergence(norm_md),
                "KF": norm_base.kl_divergence(norm_kf),
            },
            "weighting": {
                "IMQ": w_imq,
                "MD": w_md,
                "KF": w_kf,
            },
            "err-final": {
                "IMQ": mean_pred_imq,
                "MD": mean_pred_md,
                "KF": mean_pred_kf,
            },
        }

    err = jnp.linspace(-5, 5, args.grid_size)
    step = args.grid_size * 1j
    domain = jnp.mgrid[-5:5:step, -5:5:step]

    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def vpif_last(errx, erry):
        return pif_last(errx, erry)

    res = vpif_last(err, err)
    res = jax.block_until_ready(res)

    data = {
        "state": np.array(state),
        "measurements-clean": np.array(measurements_clean),
        "measurements": np.array(measurements),
        "xout": np.array(xout),
        "is_outlier": np.array(jnp.isin(jnp.arange(args.n_steps), xout)),
        "hist": {
            "KF": {
                "mean": np.array(hist_kf["mean"]),
                "cov": np.array(hist_kf["cov"]),
            },
            "WLF-IMQ": {
                "mean": np.array(hist_imq["mean"]),
                "cov": np.array(hist_imq["cov"]),
            },
        },
        "pif_grid": jax.tree.map(np.array, res["kld"]),
        "grid_axis": np.array(err),
        "domain": np.array(domain),
    }

    with args.output.open("wb") as f:
        pickle.dump(data, f)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
