#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from rebayes_mini import callbacks
from rebayes_mini.methods import gauss_filter as gf
from rebayes_mini.methods import robust_filter as rfilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gamma-process filtering experiment and store results.")
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--output", type=Path, default=Path("results/gamma-process.pkl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)

    def step(sprev, key_i):
        key_next, key_shock, key_outlier, key_obs = jax.random.split(key_i, 4)
        shock = jax.random.exponential(key_shock)
        snext = jax.random.normal(key_next) * shock * 0.01 + sprev

        is_outlier = jax.random.uniform(key_outlier) < 0.01
        obs_clean = jnp.exp(snext)
        obs = obs_clean * (1 - is_outlier) + 2 * is_outlier
        obs = obs + jax.random.gamma(key_obs, a=1 / 2)

        out = {"s": snext, "obs_clean": obs_clean, "obs": obs}
        return snext, out

    keys = jax.random.split(key, args.n_steps)
    _, hist = jax.lax.scan(step, 0.01, keys)

    def latent_fn(x):
        return x

    def measurement_fn(latent, x):
        return latent

    y = hist["obs"][:, None]
    x = jnp.ones_like(y)

    agent_kf = gf.ExtendedKalmanFilter(
        latent_fn,
        measurement_fn,
        dynamics_covariance=1.0,
        observation_covariance=1.0 * jnp.eye(1),
    )
    bel_kf = agent_kf.init_bel(0.0, cov=1.0)
    _, yhat_kf = agent_kf.scan(bel_kf, y, x[:, None], callback_fn=callbacks.get_updated_mean)

    agent_imq = rfilter.ExtendedFilterIMQ(
        mean_fn=measurement_fn,
        cov_fn=lambda _: 1.0 * jnp.eye(1),
        dynamics_covariance=1.0,
        soft_threshold=0.1,
    )
    bel_imq = agent_imq.init_bel(1.0, cov=1.0)
    _, yhat_imq = agent_imq.scan(bel_imq, y, x[:, None], callback_fn=callbacks.get_updated_mean)

    agent_iw = rfilter.ExtendedFilterInverseWishart(
        mean_fn=measurement_fn,
        dynamics_covariance=1.0,
        prior_observation_covariance=1.0 * jnp.eye(1),
        n_inner=3,
        noise_scaling=0.1,
    )
    bel_iw = agent_iw.init_bel(1.0, cov=1.0)
    _, yhat_iw = agent_iw.scan(bel_iw, y, x[:, None], callback_fn=callbacks.get_updated_mean)

    data = {
        "obs": np.array(y.squeeze()),
        "state": np.array(jnp.exp(hist["s"])),
        "pred": {
            "KF": np.array(yhat_kf).squeeze(),
            "WLF-IMQ": np.array(yhat_imq).squeeze(),
            "KF-IW": np.array(yhat_iw).squeeze(),
        },
        "meta": {"n_steps": args.n_steps, "seed": args.seed},
    }

    with args.output.open("wb") as f:
        pickle.dump(data, f)

    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
