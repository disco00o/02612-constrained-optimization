import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import sparse, optimize
from generate_test_problem import generate_test_problem
from primal_active_set import primal_active_set


# generate a new random test problem
n = 100
H, g, A, b_lower, b_upper, x_lower, x_upper = generate_test_problem(n)

# or

# load an already generated test problem
# H = sparse.load_npz("test_problems/test_data_H_size_100.npz")
# A = sparse.load_npz("test_problems/test_data_A_size_100.npz")
# g, b_lower, b_upper, x_lower, x_upper = np.load("test_problems/test_problem_size_100.npz").values()


# test our primal active set implementation
t1 = time.time()
x, lambda_, obj_val, xs = primal_active_set(H, g, A, b_lower, b_upper, x_lower, x_upper)
t2 = time.time()

print("\n ---- Results for primal active set implementation ----")
# print(f"Minimizer: {x}")
print(f"Minimum function value: {obj_val}")
print(f"Number of iterations: {len(xs)-1}")
print(f"Execution time: {t2-t1} seconds")


# solve the same problem using a scipy solver to document that we obtain the same solution
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

constraints = [
    {"type": "ineq", "fun": c1, "args": (A, b_lower)},
    {"type": "ineq", "fun": c2, "args": (A, b_upper)},
    {"type": "ineq", "fun": c3, "args": (x_lower,)},
    {"type": "ineq", "fun": c4, "args": (x_upper,)}
]

t1 = time.time()
sol = optimize.minimize(fun=func, x0=np.zeros(len(x_lower)), args=(H, g), method="SLSQP", constraints=constraints)
t2 = time.time()

print("\n ---- Results for scipy solver ----")
# print(f"Minimizer: {sol.x}")
print(f"Minimum function value: {sol.fun}")
print(f"Number of iterations: {sol.nit}")
print(f"Execution time: {t2-t1} seconds")
