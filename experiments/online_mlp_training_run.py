#!/usr/bin/env python3

import argparse
import pickle
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rebayes_mini.methods import gauss_filter as gf
from rebayes_mini.methods import replay_sgd
from rebayes_mini.methods import robust_filter as rfilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run online MLP filtering experiment and store results.")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--n-obs", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("results/online-mlp-training.pkl"))
    return parser.parse_args()


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(10)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


def f_true(x):
    c1, c2, c3, c4 = 1 / 5, -10, 1.0, 1.0
    return c1 * x + c2 * jnp.cos(c3 * jnp.pi * x) + c4 * x**3


def sample_observations_grid(key, n_obs, xmin, xmax, y_noise=3.0, p_corrupt=0.15, ycorr_max=100):
    key_x, key_y, key_shuffle, key_corrupted = jax.random.split(key, 4)
    key_cchoice, key_cval = jax.random.split(key_corrupted)

    is_corrupted = jax.random.bernoulli(key_cchoice, p=p_corrupt, shape=(n_obs,))
    ycorr = jax.random.uniform(key_cval, (n_obs,), minval=-ycorr_max, maxval=ycorr_max)
    y_noise_term = jax.random.normal(key_y, (n_obs,)) * y_noise

    x = jax.random.uniform(key_x, (n_obs,), minval=xmin, maxval=xmax)
    y = f_true(x) + y_noise_term
    y = y * (1 - is_corrupted) + ycorr * is_corrupted

    ixs_sort = jnp.argsort(x)
    return x[ixs_sort], y[ixs_sort]


def callback_ekf(bel, bel_pred, y, x, applyfn):
    return applyfn(bel_pred.mean, x[None])


def callback_filter(bel, bel_pred, y, x, applyfn):
    return applyfn(bel_pred, x[None])


def callback_ogd(bel, bel_pred, y, x, applyfn):
    return applyfn(bel_pred.mean, x[None])


def lossfn(params, counter, x, y, applyfn):
    yhat = applyfn(params, x)
    return jnp.sum(counter * (y - yhat) ** 2) / counter.sum()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    key_train, key_sample = jax.random.split(key)

    x, y = sample_observations_grid(key_sample, args.n_obs, -3, 3)
    xtest = jnp.linspace(x.min(), x.max(), args.n_obs)
    ytest = f_true(xtest)

    model = MLP()
    params_init = model.init(key_train, x[:, None])
    n_params = sum(np.size(v) for v in jax.tree_util.tree_leaves(params_init))

    Q = jnp.eye(n_params) * 1e-4
    R = 1.0 * jnp.eye(1)

    def latent_fn(z):
        return z

    results = {}

    agent_ekf = gf.ExtendedKalmanFilter(latent_fn, model.apply, dynamics_covariance=Q, observation_covariance=R)
    bel0 = agent_ekf.init_bel(params_init, cov=0.3)
    bel_ekf, yhat_pp = agent_ekf.scan(bel0, y, x[:, None], callback_fn=partial(callback_ekf, applyfn=agent_ekf.vobs_fn))
    results["EKF"] = {
        "one_step": np.array(yhat_pp).squeeze(),
        "test_pred": np.array(agent_ekf.vobs_fn(bel_ekf.mean, xtest[:, None])).squeeze(),
    }

    agent_iw = rfilter.ExtendedFilterInverseWishart(
        mean_fn=model.apply,
        dynamics_covariance=Q,
        prior_observation_covariance=R,
        noise_scaling=0.2,
        n_inner=2,
    )
    bel0 = agent_iw.init_bel(params_init, cov=0.3)
    bel_iw, yhat_pp = agent_iw.scan(bel0, y, x[:, None], callback_fn=partial(callback_filter, applyfn=agent_iw.predict_fn))
    results["EKF-IW"] = {
        "one_step": np.array(yhat_pp).squeeze(),
        "test_pred": np.array(jax.vmap(lambda x1: agent_iw.predict_fn(bel_iw, x1[None]))(xtest)).squeeze(),
    }

    agent_imq = rfilter.ExtendedFilterIMQ(
        mean_fn=model.apply,
        cov_fn=lambda _: R,
        dynamics_covariance=Q,
        soft_threshold=3.0,
    )
    bel0 = agent_imq.init_bel(params_init, cov=0.3)
    bel_imq, yhat_pp = agent_imq.scan(bel0, y, x[:, None], callback_fn=partial(callback_filter, applyfn=agent_imq.predict_fn))
    results["WLF-IMQ"] = {
        "one_step": np.array(yhat_pp).squeeze(),
        "test_pred": np.array(jax.vmap(lambda x1: agent_imq.predict_fn(bel_imq, x1[None]))(xtest)).squeeze(),
    }

    agent_md = rfilter.ExtendedFilterMD(
        mean_fn=model.apply,
        cov_fn=lambda _: R,
        dynamics_covariance=Q,
        threshold=3.0,
    )
    bel0 = agent_md.init_bel(params_init, cov=0.3)
    bel_md, yhat_pp = agent_md.scan(bel0, y, x[:, None], callback_fn=partial(callback_filter, applyfn=agent_md.predict_fn))
    results["WLF-MD"] = {
        "one_step": np.array(yhat_pp).squeeze(),
        "test_pred": np.array(jax.vmap(lambda x1: agent_md.predict_fn(bel_md, x1[None]))(xtest)).squeeze(),
    }

    agent_b = rfilter.ExtendedFilterBernoulli(
        mean_fn=model.apply,
        cov_fn=lambda _: R,
        dynamics_covariance=Q,
        alpha=1.0,
        beta=1.0,
        tol_inlier=1e-7,
        n_inner=2,
    )
    bel0 = agent_b.init_bel(params_init, cov=0.3)
    bel_b, yhat_pp = agent_b.scan(bel0, y, x[:, None], callback_fn=partial(callback_filter, applyfn=agent_b.predict_fn))
    results["EKF-B"] = {
        "one_step": np.array(yhat_pp).squeeze(),
        "test_pred": np.array(jax.vmap(lambda x1: agent_b.predict_fn(bel_b, x1[None]))(xtest)).squeeze(),
    }

    agent_ogd = replay_sgd.FifoSGD(
        model.apply,
        lossfn,
        optax.adam(1e-2),
        buffer_size=1,
        dim_features=1,
        dim_output=1,
        n_inner=2,
    )
    bel0 = agent_ogd.init_bel(params_init)
    bel_ogd, yhat_pp = agent_ogd.scan(bel0, y, x[:, None], callback_fn=partial(callback_ogd, applyfn=agent_ogd.apply_fn))
    results["OGD"] = {
        "one_step": np.array(yhat_pp).squeeze(),
        "test_pred": np.array(agent_ogd.apply_fn(bel_ogd.mean, xtest[:, None])).squeeze(),
    }

    data = {
        "x": np.array(x),
        "y": np.array(y),
        "xtest": np.array(xtest),
        "ytest": np.array(ytest),
        "results": results,
        "meta": {"seed": args.seed, "n_obs": args.n_obs},
    }

    with args.output.open("wb") as f:
        pickle.dump(data, f)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
