#!/usr/bin/env python3

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from rebayes_mini import callbacks
from rebayes_mini.methods import robust_filter as rkf


@dataclass
class LinearSSMFilterResult:
    mean: jax.Array
    cov: jax.Array


class LinearSSMRobustFilters:
    """Reusable linear-SSM filters with latent dynamics for KF/IMQ/MD variants."""

    def __init__(self, transition_matrix, projection_matrix, observation_covariance, dynamics_covariance):
        self.F = transition_matrix
        self.H = projection_matrix
        self.R = observation_covariance
        self.dynamics_covariance = dynamics_covariance
        self.I = jnp.eye(self.F.shape[0])

        if jnp.ndim(dynamics_covariance) == 0:
            self.Q = dynamics_covariance * self.I
        else:
            self.Q = dynamics_covariance

    def _measurement_fn(self, latent, _):
        return self.H @ latent

    def scan_kf(self, measurements, mean0, cov0):
        def _step(carry, y):
            mean, cov = carry
            mean_pred = self.F @ mean
            cov_pred = self.F @ cov @ self.F.T + self.Q
            err = y - self.H @ mean_pred
            s = self.H @ cov_pred @ self.H.T + self.R
            k_gain = cov_pred @ self.H.T @ jnp.linalg.inv(s)
            mean_up = mean_pred + k_gain @ err
            cov_up = (self.I - k_gain @ self.H) @ cov_pred @ (self.I - k_gain @ self.H).T + k_gain @ self.R @ k_gain.T
            out = {
                "mean": mean_up,
                "cov": cov_up,
                "weight": 1.0,
                "mean_pred": mean_pred,
            }
            return (mean_up, cov_up), out

        (meanT, covT), hist = jax.lax.scan(_step, (mean0, cov0), measurements)
        return LinearSSMFilterResult(mean=meanT, cov=covT), hist

    def scan_imq(self, measurements, mean0, cov0, soft_threshold):
        def _step(carry, y):
            mean, cov = carry
            mean_pred = self.F @ mean
            cov_pred = self.F @ cov @ self.F.T + self.Q
            err = y - self.H @ mean_pred
            weight = soft_threshold**2 / (soft_threshold**2 + jnp.inner(err, err))
            weight = jnp.maximum(weight, 1e-6)
            r_weighted = self.R / weight
            s = self.H @ cov_pred @ self.H.T + r_weighted
            k_gain = cov_pred @ self.H.T @ jnp.linalg.inv(s)
            mean_up = mean_pred + k_gain @ err
            cov_up = (self.I - k_gain @ self.H) @ cov_pred @ (self.I - k_gain @ self.H).T + k_gain @ r_weighted @ k_gain.T
            out = {
                "mean": mean_up,
                "cov": cov_up,
                "weight": weight,
                "mean_pred": mean_pred,
            }
            return (mean_up, cov_up), out

        (meanT, covT), hist = jax.lax.scan(_step, (mean0, cov0), measurements)
        return LinearSSMFilterResult(mean=meanT, cov=covT), hist

    def scan_md(self, measurements, mean0, cov0, threshold):
        r_inv = jnp.linalg.inv(self.R)

        def _step(carry, y):
            mean, cov = carry
            mean_pred = self.F @ mean
            cov_pred = self.F @ cov @ self.F.T + self.Q
            err = y - self.H @ mean_pred
            mdist = jnp.sqrt(jnp.einsum("i,ij,j->", err, r_inv, err))
            weight = (mdist < threshold).astype(mean.dtype)
            weight = jnp.maximum(weight, 1e-6)
            r_weighted = self.R / weight
            s = self.H @ cov_pred @ self.H.T + r_weighted
            k_gain = cov_pred @ self.H.T @ jnp.linalg.inv(s)
            mean_up = mean_pred + k_gain @ err
            cov_up = (self.I - k_gain @ self.H) @ cov_pred @ (self.I - k_gain @ self.H).T + k_gain @ r_weighted @ k_gain.T
            out = {
                "mean": mean_up,
                "cov": cov_up,
                "weight": weight,
                "mean_pred": mean_pred,
            }
            return (mean_up, cov_up), out

        (meanT, covT), hist = jax.lax.scan(_step, (mean0, cov0), measurements)
        return LinearSSMFilterResult(mean=meanT, cov=covT), hist

    def scan_iw(self, measurements, mean0, cov0, noise_scaling, n_inner):
        n_inner = int(n_inner)
        nsteps = len(measurements)
        agent = rkf.ExtendedFilterInverseWishart(
            mean_fn=self._measurement_fn,
            dynamics_covariance=self.dynamics_covariance,
            prior_observation_covariance=self.R,
            n_inner=n_inner,
            noise_scaling=noise_scaling,
        )
        init_bel = agent.init_bel(mean0, cov=cov0)
        x_steps = jnp.repeat(self.H[None, ...], nsteps, axis=0)
        _, hist_mean = agent.scan(
            init_bel,
            measurements,
            x_steps,
            callback_fn=callbacks.get_updated_mean,
        )
        return hist_mean

    def scan_bernoulli(self, measurements, mean0, cov0, alpha, beta, n_inner):
        n_inner = int(n_inner)
        nsteps = len(measurements)
        agent = rkf.ExtendedFilterBernoulli(
            mean_fn=self._measurement_fn,
            cov_fn=lambda _: self.R,
            dynamics_covariance=self.dynamics_covariance,
            alpha=alpha,
            beta=beta,
            tol_inlier=1e-7,
            n_inner=n_inner,
        )
        init_bel = agent.init_bel(mean0, cov=cov0)
        _, hist_mean = agent.scan(
            init_bel,
            measurements,
            jnp.ones(nsteps),
            callback_fn=callbacks.get_updated_mean,
        )
        return hist_mean
