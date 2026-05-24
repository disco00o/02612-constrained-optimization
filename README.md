# 02612-constrained-optimization

This repository is a shared space for our group to work on the 2026 exam assignment in 02612 Constrained Optimization.

The overleaf can be found [here](https://www.overleaf.com/project/69c280c0811bcf8e1975caab).



## Exercise1-EQP

The `Exercise1-EQP` folder contains MATLAB code for the equality-constrained quadratic programming exercise.

- `EQP_script.m` --> Timing and Comparison of the different EQP-solvers
- `EqualityQPSolver.m`
- `EqualityQPSolverLDLdense.m`
- `EqualityQPSolverLDLsparse.m`
- `EqualityQPSolverLUdense.m`
- `EqualityQPSolverLUsparse.m`
- `EqualityQPSolverPlainInverse.m`
- `RandomEQP.m`
- `sparseEQP_solver_comparison.m` --> Timing and Comparison of the different EQP-solvers on week 5 problem
- `Week5_ProblemGenerator.m`

## Exercise2-QP

The `Exercise2-QP` folder contains Python code for the quadratic programming exercise.

- `CPU_time_and_iteration_plots.py`
- `driverfile1.py`
- `driverfile2.py`
- `generate_test_problem.py`
- `helper_functions.py`
- `library_solvers.py`
- `primal_active_set.py`
- `primal_dual_interior_point.py`
- `test_problems/`
- `figures/`

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

## Exercise4-NLP

The `Exercise4-NLP` folder contains Python code for the nonlinear programming exercise.

- `A_Lagrangian.py`
- `B_ First_Order_Optimality_Conditions.py`
- `C_Second_Order_Optimality_Conditions.py`
- `D_SQP_Algorithm.py`
- `E_Himmelblau.py`
- `F_Himmelblau_Solve.py`
- `himmelblau.py` --> g) & h)
- `rosenbrock.py` --> i)
- `solvers.py`
- `SQP.py` --> Custom solver and associated functions
- `primal_dual_interior_point.py` --> From Exercise 2
- `helper_functions.py`
- `figures/`