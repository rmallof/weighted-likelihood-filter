# Weighted likelihood filter

![intro-plot](https://github.com/gerdm/weighted-likelihood-filter/assets/4108759/f32c0b01-433b-46e6-9153-262e1b6c4f10)

## Citation

```bib
@inproceedings{duran2024outlier,
  title={Outlier-robust Kalman filtering through generalised Bayes},
  author={Duran-Martin, Gerardo and Altamirano, Matias and Shestopaloff, Alexander Y and S{\'a}nchez-Betancourt, Leandro and Knoblauch, Jeremias and Jones, Matt and Briol, Fran{\c{c}}ois-Xavier and Murphy, Kevin},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  pages={12138--12171},
  year={2024}
}
```

## Installation
To run the experiments, make sure to have installed `jax>=0.4.2`,
[rebayes-mini](https://github.com/gerdm/rebayes-mini/tree/main),
[flax](https://github.com/google/flax),
and the [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization) package:

```bash
pip install git+https://github.com/gerdm/rebayes-mini.git
pip install flax
pip install bayesian-optimization
```

## Script workflow (2D tracking outliers)

The 2D tracking experiment is now available as standalone scripts:

- Run full experiment and store results:

```bash
cd experiments
python 2d_tracking_outlier_run.py --kind both --n-steps 1000 --n-samples 500 --init-points 20 --n-iter 30
```

This writes result files in `experiments/results`:

- `2d-ssm-outlier-mean.pkl`
- `2d-ssm-outlier-covariance.pkl`

- Generate plots from saved results:

```bash
cd experiments
python 2d_tracking_outlier_plot.py --kinds mean covariance
```

This writes figures in `experiments/figures`:

- `2d-ssm-comparison-outlier-mean.png`
- `2d-ssm-comparison-outlier-covariance.png`
- `2d-ssm-comparison-single-run-mean.png`
- `2d-ssm-comparison-single-run-covariance.png`
- `2d-ssm-comparison-outlier-both.png`

## Script workflow (all converted notebook experiments)

Each experiment now follows the same pattern:

1. Run script writes a `.pkl` result file to `experiments/results`.
2. Plot script reads the `.pkl` and writes figures to `experiments/figures`.

### Gamma process

```bash
cd experiments
python gamma_process_run.py --n-steps 2000
python gamma_process_plot.py
```

### Intro figure

```bash
cd experiments
python intro_fig_run.py --n-steps 20 --grid-size 150
python intro_fig_plot.py
```

### Online MLP training

```bash
cd experiments
python online_mlp_training_run.py --n-obs 500
python online_mlp_training_plot.py
```

### UCI regression with outliers

```bash
cd experiments
python uci_regression_outliers_run.py --dataset kin8nm --noise-type target --n-runs 100 --p-error 0.10
python uci_regression_outliers_plot.py
```

### Run all full experiments end to end

```bash
cd experiments
python gamma_process_run.py --n-steps 2000
python gamma_process_plot.py

python intro_fig_run.py --n-steps 20 --grid-size 150
python intro_fig_plot.py

python online_mlp_training_run.py --n-obs 500
python online_mlp_training_plot.py

python uci_regression_outliers_run.py --dataset kin8nm --noise-type target --n-runs 100 --p-error 0.10
python uci_regression_outliers_plot.py

python 2d_tracking_outlier_run.py --kind both --n-steps 1000 --n-samples 500 --init-points 20 --n-iter 30
python 2d_tracking_outlier_plot.py --kinds mean covariance
```
