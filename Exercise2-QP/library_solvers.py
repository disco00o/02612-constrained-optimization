import numpy as np
import time
from scipy import optimize, sparse
from generate_test_problem import generate_test_problem
from qpsolvers import Problem, solve_problem

# load test problem
H = sparse.load_npz("test_problems/test_data_H_size_100.npz")
A = sparse.load_npz("test_problems/test_data_A_size_100.npz")
g, b_lower, b_upper, x_lower, x_upper = np.load("test_problems/test_problem_size_100.npz").values()


### scipy SLSQP
def func(x, H, g):
    return 1/2 * x.T @ H @ x + g @ x

# constraints must be of the form >= 0
def c1(x, A, b_lower):
    return A.T @ x - b_lower

def c2(x, A, b_upper):
    return -(A.T @ x) + b_upper

def c3(x, x_lower):
    return x - x_lower

def c4(x, x_upper):
    return -x + x_upper

constraints = [
    {"type": "ineq", "fun": c1, "args": (A, b_lower)},
    {"type": "ineq", "fun": c2, "args": (A, b_upper)},
    {"type": "ineq", "fun": c3, "args": (x_lower,)},
    {"type": "ineq", "fun": c4, "args": (x_upper,)}
]

# initial guess
x0 = np.zeros(len(x_lower))

t1 = time.time()
sol = optimize.minimize(fun=func, x0=x0, args=(H, g), method="SLSQP", constraints=constraints)
t2 = time.time()

print("----- SLSQP -----")
print(f"Minimum function value: {sol.fun}")
print(f"Number of iterations: {sol.nit}")
print(f"Time: {t2-t1} seconds")

print("\n")

### qpsolvers OSQP
G = sparse.vstack([A.T, -A.T]).tocsc()
h = np.concatenate((b_upper, -b_lower))

problem = Problem(P=H, q=g, G=G, h=h, lb=x_lower, ub=x_upper)
t3 = time.time()
sol2 = solve_problem(problem, solver="osqp")
t4 = time.time()

osqp_info = sol2.extras["info"]
num_iters = osqp_info.iter
minimizer = osqp_info.obj_val
 
print("----- OSQP -----")
print(f"Minimum function value: {minimizer}")
print(f"Number of iterations: {num_iters}")
print(f"Time: {t4-t3} seconds")

