# When does the Gaussian score baseline work?

Median relative score error of the analytic Gaussian baseline (== covariance-matched Gaussian model) vs fine-grid BP.

| family | kurt. | rho | small t (min) | mid t (0.2) | large t (max) |
|---|---|---|---|---|---|
| gauss_mix_kappa0.3 | -0.18 | 0.5 | 0.060 | 0.008 | 0.000 |
| gauss_mix_kappa0.6 | -0.72 | 0.5 | 0.305 | 0.031 | 0.000 |
| gauss_mix_kappa0.9 | -1.62 | 0.5 | 0.792 | 0.077 | 0.000 |
| laplace | +3.00 | 0.5 | 0.402 | 0.070 | 0.000 |
| student_t_nu5 | +6.00 | 0.5 | 0.272 | 0.057 | 0.000 |
| student_t_nu8 | +1.50 | 0.5 | 0.164 | 0.031 | 0.000 |
| gauss_mix_kappa0.3 | -0.18 | 0.85 | 0.034 | 0.002 | 0.000 |
| gauss_mix_kappa0.6 | -0.72 | 0.85 | 0.149 | 0.009 | 0.000 |
| gauss_mix_kappa0.9 | -1.62 | 0.85 | 0.520 | 0.028 | 0.000 |
| laplace | +3.00 | 0.85 | 0.258 | 0.031 | 0.000 |
| student_t_nu5 | +6.00 | 0.85 | 0.167 | 0.025 | 0.000 |
| student_t_nu8 | +1.50 | 0.85 | 0.098 | 0.012 | 0.000 |
| gauss_mix_kappa0.3 | -0.18 | 0.95 | 0.010 | 0.001 | 0.000 |
| gauss_mix_kappa0.6 | -0.72 | 0.95 | 0.052 | 0.003 | 0.000 |
| gauss_mix_kappa0.9 | -1.62 | 0.95 | 0.132 | 0.009 | 0.000 |
| laplace | +3.00 | 0.95 | 0.113 | 0.012 | 0.000 |
| student_t_nu5 | +6.00 | 0.95 | 0.081 | 0.011 | 0.000 |
| student_t_nu8 | +1.50 | 0.95 | 0.041 | 0.004 | 0.000 |