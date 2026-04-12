# 02612-constrained-optimization

This repository is a shared space for our group to work on the 2026 exam assignment in 02612 Constrained Optimization.

The overleaf can be found [here](https://www.overleaf.com/project/69c280c0811bcf8e1975caab).

## Exercise3-LP

The `Exercise3-LP` folder contains MATLAB code for the LP exercise.

- **`Question3/RandomLP.m`**: Generates random test instances
- **`Question4/LP_linprog.m`**: Solves a random LP with `linprog`'s dual-simplex algorithm
- **`Question6/`**: 
  - **`LPippd.m`** implements a primal-dual interior-point method for solving standard-form LPs
  - **`LPStandardForm.m`**: converts bounded LPs into standard form and solves them  using **`LPippd.m`**
  - **`LPippd_test.m`**: solves a random LP test instance with the custom interior-point implementation **`LPStandardForm.m`**
- **`Question8/`**: 
  - **`revised_simplex.m`** implements a revised simplex method for solving standard-form LPs
  - **`solve_lp.m`**: converts bounded LPs into standard form and solves them using a two-phase revised simplex approach calling twice for **`revised_simplex.m`**
  - **`test_simplex.m`**: solves a random LP test instance with the custom revised simplex implementation **`solve_lp.m`**
- **`Question9/comparison.m`**: compares solver performance between `linprog` (dual-simplex and interior-point), the custom interior-point solver and the custom revised simplex implementation
- **`Figures/`**: A directory containing all visualizations used in the report