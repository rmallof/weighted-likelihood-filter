#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
from time import time

import datagen
import jax
import jax.numpy as jnp
import numpy as np
from bayes_opt import BayesianOptimization
from tqdm import tqdm

from ssm_robust_filters import LinearSSMRobustFilters


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run 2D tracking with outliers and store posterior trajectories."
    )
    parser.add_argument(
        "--kind",
        choices=["mean", "covariance", "added_mean", "both"],
        default="both",
        help="Outlier model to run.",
    )
    parser.add_argument("--n-steps", type=int, default=200, help="Trajectory length.")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of Monte Carlo runs.")
    parser.add_argument("--seed", type=int, default=314, help="Random seed.")
    parser.add_argument("--init-points", type=int, default=5, help="BayesOpt init points.")
    parser.add_argument("--n-iter", type=int, default=5, help="BayesOpt optimization iterations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "results",
        help="Directory where result .pkl files are stored.",
    )
    return parser.parse_args()


def build_generator(kind: str, delta: float, dynamics_covariance: float, obs_covariance: float):
    if kind == "covariance":
        return datagen.GaussStMovingObject2D(
            delta,
            dynamics_covariance,
            obs_covariance,
            dof_observed=2.01,
        )
    if kind == "mean":
        return datagen.GaussMeanOutlierMovingObject2D(
            delta,
            dynamics_covariance,
            obs_covariance,
            outlier_proba=0.05,
            outlier_scale=2.0,
        )
    if kind == "added_mean":
        return datagen.GaussOneSideOutlierMovingObject2D(
            delta,
            dynamics_covariance,
            obs_covariance,
            outlier_proba=0.05,
            outlier_minval=-100,
            outlier_maxval=100,
        )
    raise ValueError(f"Unsupported kind: {kind}")


def run_experiment(kind: str, args: argparse.Namespace) -> dict:
    delta = 0.1
    dynamics_covariance = 0.1
    obs_covariance = 10.0

    dgen = build_generator(kind, delta, dynamics_covariance, obs_covariance)

    key = jax.random.PRNGKey(args.seed)
    initial_mean = jnp.array([0.0, 0.0, 1.0, 1.0])
    keys = jax.random.split(key, args.n_samples)
    datasets = jax.vmap(dgen.sample, in_axes=(0, None, None))(keys, initial_mean, args.n_steps)

    yv = datasets["observed"]
    statev = datasets["latent"]

    ssm_filters = LinearSSMRobustFilters(
        transition_matrix=dgen.transition_matrix,
        projection_matrix=dgen.projection_matrix,
        observation_covariance=dgen.observation_covariance,
        dynamics_covariance=dgen.dynamics_covariance,
    )

    time_methods = {}
    hist_methods = {}
    configs = {}

    @jax.jit
    def filter_kf(measurements, state):
        bel0_cov = jnp.eye(initial_mean.shape[0])
        _, hist = ssm_filters.scan_kf(measurements, initial_mean, bel0_cov)
        hist_mean = hist["mean"]
        err = jnp.sqrt(jnp.power(hist_mean - state, 2).sum(axis=0))
        return err, hist_mean

    def filter_kfiw(noise_scaling, n_inner, measurements, state):
        hist = ssm_filters.scan_iw(
            measurements,
            initial_mean,
            1.0,
            noise_scaling,
            n_inner,
        )
        err = jnp.sqrt(jnp.power(hist - state, 2).sum(axis=0))
        return err, hist

    def bo_filter_kfiw(noise_scaling, n_inner):
        err, _ = filter_kfiw(noise_scaling, n_inner, yv[0], statev[0])
        err = err.max()
        err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
        return -err

    @jax.jit
    def filter_wlfimq(soft_threshold, measurements, state):
        bel0_cov = jnp.eye(initial_mean.shape[0])
        _, hist = ssm_filters.scan_imq(measurements, initial_mean, bel0_cov, soft_threshold)
        hist_mean = hist["mean"]
        err = jnp.sqrt(jnp.power(hist_mean - state, 2).sum(axis=0))
        return err, hist_mean

    @jax.jit
    def bo_filter_wlfimq(soft_threshold):
        err, _ = filter_wlfimq(soft_threshold, yv[0], statev[0])
        return -err.max()

    @jax.jit
    def filter_wlfmd(threshold, measurements, state):
        bel0_cov = jnp.eye(initial_mean.shape[0])
        _, hist = ssm_filters.scan_md(measurements, initial_mean, bel0_cov, threshold)
        hist_mean = hist["mean"]
        err = jnp.sqrt(jnp.power(hist_mean - state, 2).sum(axis=0))
        return err, hist_mean

    @jax.jit
    def bo_filter_wlfmd(threshold):
        err, _ = filter_wlfmd(threshold, yv[0], statev[0])
        return -err.max()

    def filter_kfb(alpha, beta, n_inner, measurements, state):
        hist = ssm_filters.scan_bernoulli(
            measurements,
            initial_mean,
            1.0,
            alpha,
            beta,
            n_inner,
        )
        err = jnp.sqrt(jnp.power(hist - state, 2).sum(axis=0))
        return err, hist

    def bo_filter_kfb(alpha, beta, n_inner):
        err, _ = filter_kfb(alpha, beta, n_inner, yv[0], statev[0])
        err = -err.max()
        err = jax.lax.cond(jnp.isnan(err), lambda: -1e6, lambda: err)
        return err

    def evaluate_fixed_hparams(method: str, run_fn, *run_args):
        hist_bel = []
        times = []
        for y, state in tqdm(zip(yv, statev), total=args.n_samples, desc=method):
            t_init = time()
            _, run = run_fn(*run_args, y, state)
            run = np.array(run)
            t_end = time()
            hist_bel.append(run)
            times.append(t_end - t_init)
        hist_methods[method] = np.stack(hist_bel)
        time_methods[method] = np.array(times)

    evaluate_fixed_hparams("KF", filter_kf)

    bo = BayesianOptimization(
        bo_filter_kfiw,
        pbounds={"noise_scaling": (1e-6, 20), "n_inner": (1, 10)},
        random_state=args.seed,
        verbose=0,
        allow_duplicate_points=True,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    cfg = bo.max["params"]
    configs["KF-IW"] = cfg
    evaluate_fixed_hparams("KF-IW", filter_kfiw, cfg["noise_scaling"], int(cfg["n_inner"]))

    bo = BayesianOptimization(
        bo_filter_wlfimq,
        pbounds={"soft_threshold": (1e-6, 20)},
        random_state=args.seed,
        verbose=0,
        allow_duplicate_points=True,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    cfg = bo.max["params"]
    configs["WoLF-IMQ"] = cfg
    evaluate_fixed_hparams("WoLF-IMQ", filter_wlfimq, cfg["soft_threshold"])

    bo = BayesianOptimization(
        bo_filter_wlfmd,
        pbounds={"threshold": (1e-6, 20)},
        random_state=args.seed,
        verbose=0,
        allow_duplicate_points=True,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    cfg = bo.max["params"]
    configs["WoLF-MD"] = cfg
    evaluate_fixed_hparams("WoLF-MD", filter_wlfmd, cfg["threshold"])

    bo = BayesianOptimization(
        bo_filter_kfb,
        pbounds={"alpha": (0.0, 5.0), "beta": (0.0, 5.0), "n_inner": (1, 10)},
        random_state=args.seed,
        verbose=0,
        allow_duplicate_points=True,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    cfg = bo.max["params"]
    configs["KF-B"] = cfg
    evaluate_fixed_hparams("KF-B", filter_kfb, cfg["alpha"], cfg["beta"], int(cfg["n_inner"]))

    return {
        "datasets": jax.tree.map(np.array, datasets),
        "time": {k: np.array(v) for k, v in time_methods.items()},
        "posterior-states": {k: np.array(v) for k, v in hist_methods.items()},
        "config": configs,
        "name": kind,
        "meta": {
            "n_steps": args.n_steps,
            "n_samples": args.n_samples,
            "seed": args.seed,
            "init_points": args.init_points,
            "n_iter": args.n_iter,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kinds = ["mean", "covariance"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        print(f"Running kind={kind}")
        data = run_experiment(kind, args)
        output_file = args.output_dir / f"2d-ssm-outlier-{kind}.pkl"
        with output_file.open("wb") as f:
            pickle.dump(data, f)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
