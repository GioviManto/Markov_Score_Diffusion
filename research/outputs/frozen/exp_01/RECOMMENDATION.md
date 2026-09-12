# Grid parameter recommendation (from exp_01 heatmap, rho = 0.95)

M =  101, A = 4: worst rel. error = 6.885e-08
M =  101, A = 5: worst rel. error = 9.273e-07
M =  101, A = 6: worst rel. error = 1.299e-04
M =  101, A = 7: worst rel. error = 2.647e-03
M =  101, A = 8: worst rel. error = 1.655e-02
M =  201, A = 4: worst rel. error = 6.513e-08
M =  201, A = 5: worst rel. error = 4.659e-14
M =  201, A = 6: worst rel. error = 8.284e-15
M =  201, A = 7: worst rel. error = 3.249e-14
M =  201, A = 8: worst rel. error = 8.808e-11
M =  401, A = 4: worst rel. error = 6.419e-08
M =  401, A = 5: worst rel. error = 4.495e-14
M =  401, A = 6: worst rel. error = 7.460e-15
M =  401, A = 7: worst rel. error = 8.417e-15
M =  401, A = 8: worst rel. error = 8.714e-15
M =  801, A = 4: worst rel. error = 6.396e-08
M =  801, A = 5: worst rel. error = 4.459e-14
M =  801, A = 6: worst rel. error = 7.589e-15
M =  801, A = 7: worst rel. error = 8.647e-15
M =  801, A = 8: worst rel. error = 7.838e-15

Cheapest configuration with worst-case error < 1e-4 over tested t: M = 101, A = 4.

Caveats for the working default:
- This calibration is Gaussian; heavy-tailed innovations (Laplace,
  Student-t) put more posterior mass in the tails, so the half-width
  should NOT be reduced to the Gaussian-optimal value. We adopt
  M = 401, A = 8 as the reference configuration for all non-Gaussian
  experiments (error floor ~1e-15 in the Gaussian calibration, cost
  O(n M^2) still negligible), and M = 801, A = 8 for convergence checks.
- The binding constraint at very small t is the likelihood width
  sqrt(Delta)/alpha ~ sqrt(2t) versus the grid step 2A/(M-1); the
  `resolution_ok` column flags settings with < 3 grid points per
  likelihood standard deviation.