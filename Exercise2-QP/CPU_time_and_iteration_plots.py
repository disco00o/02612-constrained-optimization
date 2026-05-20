import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import sparse, optimize
from generate_test_problem import generate_test_problem
from primal_active_set import primal_active_set
from primal_dual_interior_point import primal_dual_interior_point
from qpsolvers import Problem, solve_problem

### SLSQP setup
def func(x, H, g):
    return 1/2 * x.T @ H @ x + g @ x

def c1(x, A, b_lower):
    return A.T @ x - b_lower

def c2(x, A, b_upper):
    return -(A.T @ x) + b_upper

def c3(x, x_lower):
    return x - x_lower

def c4(x, x_upper):
    return -x + x_upper


problem_sizes = [10, 50, 100, 150, 200, 250, 300]
# problem_sizes = [10, 50, 100]

SLSQP_cpu = []
SLSQP_iter = []

OSQP_cpu = []
OSQP_iter = []

active_set_cpu = []
active_set_iter = []

interior_point_cpu = []
interior_point_iter = []

for n in problem_sizes:
    
    H, g, A, b_lower, b_upper, x_lower, x_upper = generate_test_problem(n)

    # SLSQP setup
    constraints = [
    {"type": "ineq", "fun": c1, "args": (A, b_lower)},
    {"type": "ineq", "fun": c2, "args": (A, b_upper)},
    {"type": "ineq", "fun": c3, "args": (x_lower,)},
    {"type": "ineq", "fun": c4, "args": (x_upper,)}
    ]

    # OSQP setup
    G = sparse.vstack([A.T, -A.T]).tocsc()
    h = np.concatenate((b_upper, -b_lower))
    problem = Problem(P=H, q=g, G=G, h=h, lb=x_lower, ub=x_upper)


    # run every algorithm and save statistics

    # SLSQP
    t1 = time.time()
    sol = optimize.minimize(fun=func, x0=np.zeros(len(x_lower)), args=(H, g), method="SLSQP", constraints=constraints)
    t2 = time.time()
    SLSQP_cpu.append(t2-t1)
    SLSQP_iter.append(sol.nit)
    print(f"SLSQP size {n} done")

    # OSQP
    t1 = time.time()
    sol = solve_problem(problem, solver="osqp")
    t2 = time.time()
    osqp_info = sol.extras["info"]
    OSQP_cpu.append(t2-t1)
    OSQP_iter.append(osqp_info.iter)
    print(f"OSQP size {n} done")

    # primal active-set
    t1 = time.time()
    x, lambda_, obj_val, xs = primal_active_set(H, g, A, b_lower, b_upper, x_lower, x_upper)
    t2 = time.time()
    active_set_cpu.append(t2-t1)
    active_set_iter.append(len(xs)-1)
    print(f"Active-set size {n} done")

    # primal-dual interior-point
    t1 = time.time()
    x, z, s, obj_val, xs, rLs, rAs, mus = primal_dual_interior_point(H, g, A, b_lower, b_upper, x_lower, x_upper, x0 = np.zeros(len(x_lower)))
    t2 = time.time()
    interior_point_cpu.append(t2-t1)
    interior_point_iter.append(len(xs)-1)
    print(f"Interior-point size {n} done")


# iteration count plot
plt.figure(figsize=(10, 6))

plt.plot(problem_sizes, SLSQP_iter, marker='o', linestyle='-', label="SLSQP")
plt.plot(problem_sizes, OSQP_iter, marker='o', linestyle='-', label="OSQP")
plt.plot(problem_sizes, active_set_iter, marker='o', linestyle='-', label="Active-set")
plt.plot(problem_sizes, interior_point_iter, marker='o', linestyle='-', label="Interior-point")

plt.xlabel('Problem size (n)', fontsize=12)
plt.ylabel('Iterations', fontsize=12)
plt.title('Algorithm iteration count comparison', fontsize=14)
plt.grid(True, which="both", linestyle=':', alpha=1)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("figures/Iteration_count_comparison.png")

# CPU time plot
plt.figure(figsize=(10, 6))

plt.plot(problem_sizes, SLSQP_cpu, marker='o', linestyle='-', label="SLSQP")
plt.plot(problem_sizes, OSQP_cpu, marker='o', linestyle='-', label="OSQP")
plt.plot(problem_sizes, active_set_cpu, marker='o', linestyle='-', label="Active-set")
plt.plot(problem_sizes, interior_point_cpu, marker='o', linestyle='-', label="Interior-point")

plt.xlabel('Problem size (n)', fontsize=12)
plt.ylabel('Seconds', fontsize=12)
plt.title('Algorithm CPU time comparison', fontsize=14)
plt.grid(True, which="both", linestyle=':', alpha=1)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("figures/CPU_time_comparison.png")

plt.yscale('log')
plt.savefig("figures/CPU_time_comparison_log.png")


