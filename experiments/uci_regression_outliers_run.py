#!/usr/bin/env python3

import argparse
import pickle
from functools import partial
from pathlib import Path
from time import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from bayes_opt import BayesianOptimization
from rebayes_mini.methods import replay_sgd
from rebayes_mini.methods import robust_filter as rkf
from tqdm import tqdm

import datagen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UCI regression outlier benchmark and store results.")
    parser.add_argument("--dataset", type=str, default="kin8nm")
    parser.add_argument("--noise-type", type=str, default="target", choices=["target", "covariate"])
    parser.add_argument("--p-error", type=float, default=0.10)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--v-error", type=float, default=50.0)
    parser.add_argument("--init-points", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("results/uci-regression-outliers.pkl"))
    return parser.parse_args()


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(20)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def callback_filter(bel, bel_pred, y, x, applyfn):
    return applyfn(bel_pred, x[None])


def callback_ogd(bel, bel_pred, y, x, applyfn):
    return applyfn(bel_pred.mean, x[None])


def lossfn(params, counter, x, y, applyfn):
    yhat = applyfn(params, x)
    return jnp.sum(counter * (y - yhat) ** 2) / counter.sum()


def build_bopt_step(filterfn, y, x):
    def opt_step(**hparams):
        yhat_pp = filterfn(**hparams, measurements=y, covariates=x)
        err = jnp.power(yhat_pp - y, 2)
        err = jnp.median(err)
        err = jax.lax.cond(jnp.isnan(err), lambda: 1e6, lambda: err)
        return -err

    return opt_step


def cast_hparams(hparams, float_keys=(), int_keys=()):
    out = dict(hparams)
    for key in float_keys:
        if key in out:
            out[key] = float(out[key])
    for key in int_keys:
        if key in out:
            out[key] = int(out[key])
    return out


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    args.output.parent.mkdir(parents=True, exist_ok=True)

    X_collection, y_collection, ix_clean_collection = datagen.create_uci_collection(
        dataset_name=args.dataset,
        noise_type=args.noise_type,
        p_error=args.p_error,
        n_runs=args.n_runs,
        v_error=args.v_error,
        seed_init=args.seed,
        path=str(base_dir / "data"),
    )

    y, x = y_collection[0], X_collection[0]

    model = MLP()
    params_init = model.init(jax.random.PRNGKey(args.seed), x[:1])

    n_params = sum(np.size(v) for v in jax.tree_util.tree_leaves(params_init))
    q = jnp.eye(n_params) * 1e-4
    observation_covariance = jnp.eye(1) * 1.0

    measurement_fn = model.apply

    time_methods = {}
    hist_methods = {}
    configs = {}

    @jax.jit
    def filter_kf(log_lr, measurements, covariates):
        lr = jnp.exp(log_lr)
        agent = rkf.ExtendedFilterIMQ(
            mean_fn=measurement_fn,
            cov_fn=lambda _: observation_covariance,
            dynamics_covariance=q,
            soft_threshold=1e8,
        )
        init_bel = agent.init_bel(params_init, cov=lr)
        callback = partial(callback_filter, applyfn=agent.predict_fn)
        _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
        return yhat_pp.squeeze()

    opt_step = build_bopt_step(filter_kf, y, x)
    bo = BayesianOptimization(
        opt_step,
        pbounds={"log_lr": (-5, 0)},
        random_state=args.seed,
        allow_duplicate_points=True,
        verbose=0,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    method = "KF"
    cfg = cast_hparams(bo.max["params"], float_keys=("log_lr",))
    log_lr = cfg["log_lr"]
    configs[method] = cfg
    hist_bel = []
    times = []
    for yc, xc in tqdm(zip(y_collection, X_collection), total=args.n_runs, desc=method):
        tinit = time()
        run = filter_kf(log_lr, y, x)
        run = jax.block_until_ready(run)
        tend = time()
        hist_bel.append(run)
        times.append(tend - tinit)
    hist_methods[method] = np.stack(hist_bel)
    time_methods[method] = times

    def filter_kfb(log_lr, alpha, beta, n_inner, measurements, covariates):
        lr = jnp.exp(jnp.array(log_lr))
        alpha = float(alpha)
        beta = float(beta)
        n_inner = int(n_inner)
        agent = rkf.ExtendedFilterBernoulli(
            mean_fn=measurement_fn,
            cov_fn=lambda _: observation_covariance,
            dynamics_covariance=q,
            alpha=alpha,
            beta=beta,
            tol_inlier=1e-7,
            n_inner=n_inner,
        )
        init_bel = agent.init_bel(params_init, cov=lr)
        callback = partial(callback_filter, applyfn=agent.predict_fn)
        _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
        return yhat_pp.squeeze()

    opt_step = build_bopt_step(filter_kfb, y, x)
    bo = BayesianOptimization(
        opt_step,
        pbounds={
            "log_lr": (-5, 0),
            "alpha": (0.0, 5.0),
            "beta": (0.0, 5.0),
            "n_inner": (1, 10),
        },
        random_state=args.seed,
        allow_duplicate_points=True,
        verbose=0,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    method = "KF-B"
    cfg = cast_hparams(bo.max["params"], float_keys=("log_lr", "alpha", "beta"), int_keys=("n_inner",))
    configs[method] = cfg
    hist_bel = []
    times = []
    for yc, xc in tqdm(zip(y_collection, X_collection), total=args.n_runs, desc=method):
        tinit = time()
        run = filter_kfb(**cfg, measurements=y, covariates=x)
        run = jax.block_until_ready(run)
        tend = time()
        hist_bel.append(run)
        times.append(tend - tinit)
    hist_methods[method] = np.stack(hist_bel)
    time_methods[method] = times

    def filter_kfiw(log_lr, noise_scaling, n_inner, measurements, covariates):
        lr = jnp.exp(jnp.array(log_lr))
        noise_scaling = float(noise_scaling)
        n_inner = int(n_inner)
        agent = rkf.ExtendedFilterInverseWishart(
            mean_fn=measurement_fn,
            dynamics_covariance=q,
            prior_observation_covariance=observation_covariance,
            n_inner=n_inner,
            noise_scaling=noise_scaling,
        )
        init_bel = agent.init_bel(params_init, cov=lr)
        callback = partial(callback_filter, applyfn=agent.predict_fn)
        _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
        return yhat_pp.squeeze()

    opt_step = build_bopt_step(filter_kfiw, y, x)
    bo = BayesianOptimization(
        opt_step,
        pbounds={"log_lr": (-5, 0), "noise_scaling": (1e-6, 20), "n_inner": (1, 10)},
        random_state=args.seed,
        allow_duplicate_points=True,
        verbose=0,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    method = "KF-IW"
    cfg = cast_hparams(bo.max["params"], float_keys=("log_lr", "noise_scaling"), int_keys=("n_inner",))
    configs[method] = cfg
    hist_bel = []
    times = []
    for yc, xc in tqdm(zip(y_collection, X_collection), total=args.n_runs, desc=method):
        tinit = time()
        run = filter_kfiw(**cfg, measurements=y, covariates=x)
        run = jax.block_until_ready(run)
        tend = time()
        hist_bel.append(run)
        times.append(tend - tinit)
    hist_methods[method] = np.stack(hist_bel)
    time_methods[method] = times

    @jax.jit
    def filter_wlfimq(log_lr, soft_threshold, measurements, covariates):
        lr = jnp.exp(log_lr)
        agent = rkf.ExtendedFilterIMQ(
            mean_fn=measurement_fn,
            cov_fn=lambda _: observation_covariance,
            dynamics_covariance=q,
            soft_threshold=soft_threshold,
        )
        init_bel = agent.init_bel(params_init, cov=lr)
        callback = partial(callback_filter, applyfn=agent.predict_fn)
        _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
        return yhat_pp.squeeze()

    opt_step = build_bopt_step(filter_wlfimq, y, x)
    bo = BayesianOptimization(
        opt_step,
        pbounds={"log_lr": (-5, 0), "soft_threshold": (1e-6, 20)},
        random_state=args.seed,
        allow_duplicate_points=True,
        verbose=0,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    method = "WLF-IMQ"
    cfg = cast_hparams(bo.max["params"], float_keys=("log_lr", "soft_threshold"))
    configs[method] = cfg
    hist_bel = []
    times = []
    for yc, xc in tqdm(zip(y_collection, X_collection), total=args.n_runs, desc=method):
        tinit = time()
        run = filter_wlfimq(**cfg, measurements=y, covariates=x)
        run = jax.block_until_ready(run)
        tend = time()
        hist_bel.append(run)
        times.append(tend - tinit)
    hist_methods[method] = np.stack(hist_bel)
    time_methods[method] = times

    @jax.jit
    def filter_wlfmd(log_lr, threshold, measurements, covariates):
        lr = jnp.exp(log_lr)
        agent = rkf.ExtendedFilterMD(
            mean_fn=measurement_fn,
            cov_fn=lambda _: observation_covariance,
            dynamics_covariance=q,
            threshold=threshold,
        )
        init_bel = agent.init_bel(params_init, cov=lr)
        callback = partial(callback_filter, applyfn=agent.predict_fn)
        _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
        return yhat_pp.squeeze()

    opt_step = build_bopt_step(filter_wlfmd, y, x)
    bo = BayesianOptimization(
        opt_step,
        pbounds={"log_lr": (-5, 0), "threshold": (1e-6, 20)},
        random_state=args.seed,
        allow_duplicate_points=True,
        verbose=0,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    method = "WLF-MD"
    cfg = cast_hparams(bo.max["params"], float_keys=("log_lr", "threshold"))
    configs[method] = cfg
    hist_bel = []
    times = []
    for yc, xc in tqdm(zip(y_collection, X_collection), total=args.n_runs, desc=method):
        tinit = time()
        run = filter_wlfmd(**cfg, measurements=y, covariates=x)
        run = jax.block_until_ready(run)
        tend = time()
        hist_bel.append(run)
        times.append(tend - tinit)
    hist_methods[method] = np.stack(hist_bel)
    time_methods[method] = times

    def filter_ogd(log_lr, n_inner, measurements, covariates):
        lr = jnp.exp(jnp.array(log_lr))
        n_inner = int(n_inner)
        agent = replay_sgd.FifoSGD(
            measurement_fn,
            lossfn,
            optax.adam(lr),
            buffer_size=1,
            dim_features=covariates.shape[-1],
            dim_output=1,
            n_inner=n_inner,
        )
        callback = partial(callback_ogd, applyfn=measurement_fn)
        init_bel = agent.init_bel(params_init)
        _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
        return yhat_pp.squeeze()

    opt_step = build_bopt_step(filter_ogd, y, x)
    bo = BayesianOptimization(
        opt_step,
        pbounds={"log_lr": (-5, 0), "n_inner": (1, 10)},
        random_state=args.seed,
        allow_duplicate_points=True,
        verbose=0,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    method = "OGD"
    cfg = cast_hparams(bo.max["params"], float_keys=("log_lr",), int_keys=("n_inner",))
    configs[method] = cfg
    hist_bel = []
    times = []
    for yc, xc in tqdm(zip(y_collection, X_collection), total=args.n_runs, desc=method):
        tinit = time()
        run = filter_ogd(**cfg, measurements=y, covariates=x)
        run = jax.block_until_ready(run)
        tend = time()
        hist_bel.append(run)
        times.append(tend - tinit)
    hist_methods[method] = np.stack(hist_bel)
    time_methods[method] = times

    def filter_wlf_ogd(log_lr, soft_threshold, n_inner, measurements, covariates):
        lr = jnp.exp(jnp.array(log_lr))
        soft_threshold = float(soft_threshold)
        n_inner = int(n_inner)
        agent = rkf.FifoSGDIMQ(
            measurement_fn,
            optax.adam(lr),
            buffer_size=1,
            dim_features=covariates.shape[-1],
            dim_output=1,
            soft_threshold=soft_threshold,
            n_inner=n_inner,
        )
        callback = partial(callback_ogd, applyfn=measurement_fn)
        init_bel = agent.init_bel(params_init)
        _, yhat_pp = agent.scan(init_bel, measurements, covariates, callback_fn=callback)
        return yhat_pp.squeeze()

    opt_step = build_bopt_step(filter_wlf_ogd, y, x)
    bo = BayesianOptimization(
        opt_step,
        pbounds={"log_lr": (-10, 0), "soft_threshold": (1e-6, 20), "n_inner": (1, 10)},
        random_state=args.seed,
        allow_duplicate_points=True,
        verbose=0,
    )
    bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    method = "WLF-OGD"
    cfg = cast_hparams(
        bo.max["params"],
        float_keys=("log_lr", "soft_threshold"),
        int_keys=("n_inner",),
    )
    configs[method] = cfg
    hist_bel = []
    times = []
    for yc, xc in tqdm(zip(y_collection, X_collection), total=args.n_runs, desc=method):
        tinit = time()
        run = filter_wlf_ogd(**cfg, measurements=y, covariates=x)
        run = jax.block_until_ready(run)
        tend = time()
        hist_bel.append(run)
        times.append(tend - tinit)
    hist_methods[method] = np.stack(hist_bel)
    time_methods[method] = times

    rmedse_df = pd.DataFrame(
        jax.tree.map(lambda z: np.sqrt(np.median(np.power(z - y_collection, 2), 1)), hist_methods)
    )
    rmedse_df = rmedse_df.reset_index().melt("index")
    rmedse_df = rmedse_df.rename({"index": "run", "variable": "method", "value": "err"}, axis=1)

    time_df = pd.DataFrame(time_methods).reset_index().melt("index")
    time_df = time_df.rename({"index": "run", "variable": "method", "value": "time"}, axis=1)

    summary_df = rmedse_df.merge(time_df, on=["method", "run"]).query("run > 0")

    data = {
        "datasets": {
            "X": np.array(X_collection),
            "y": np.array(y_collection),
        },
        "time": {k: np.array(v) for k, v in time_methods.items()},
        "posterior-states": hist_methods,
        "config": configs,
        "dataset-name": args.dataset,
        "noise-type": args.noise_type,
        "p-error": args.p_error,
        "summary": {
            "rmedse_median": summary_df.groupby("method")["err"].median().to_dict(),
            "time_median": summary_df.groupby("method")["time"].median().to_dict(),
        },
        "meta": {
            "n-runs": args.n_runs,
            "seed": args.seed,
            "v-error": args.v_error,
            "init-points": args.init_points,
            "n-iter": args.n_iter,
            "uses-first-run-data-for-eval-loop": True,
        },
    }

    with args.output.open("wb") as f:
        pickle.dump(data, f)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
