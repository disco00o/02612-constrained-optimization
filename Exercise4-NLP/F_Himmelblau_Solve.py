import numpy as np
import time
from scipy.optimize import minimize

def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

def grad_himmelblau(x):
    x1, x2 = x
    df_dx1 = 4 * x1 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7)
    df_dx2 = 2 * (x1**2 + x2 - 11) + 4 * x2 * (x1 + x2**2 - 7)
    return np.array([df_dx1, df_dx2])

constraints = [
    {'type': 'ineq', 'fun': lambda x: (x[0] + 2)**2 - x[1]},
    {'type': 'ineq', 'fun': lambda x: -4 * x[0] + 10 * x[1]}
]

points = []

t1 = time.time()
res = minimize(
            himmelblau,
            np.array([0, 0]),
            method='SLSQP',
            jac=grad_himmelblau,
            constraints=constraints)
t2 = time.time()

print(f"x = {res.x}, f(x) = {himmelblau(res.x):.6f}")
print(f"Number of iterations: {res.nit}")
print(f"Time: {t2-t1} seconds")
