import numpy as np
from scipy.optimize import minimize, Bounds


def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def grad_himmelblau(x):
    x1, x2 = x

    return np.array([
        4 * x1 * (x1**2 + x2 - 11) + 2 * (x1 + x2**2 - 7),
        2 * (x1**2 + x2 - 11) + 4 * x2 * (x1 + x2**2 - 7)
    ])


def hess_himmelblau(x):
    x1, x2 = x

    return np.array([
        [12 * x1**2 + 4 * x2 - 42, 4 * x1 + 4 * x2],
        [4 * x1 + 4 * x2, 4 * x1 + 12 * x2**2 - 26]
    ])


if __name__ == "__main__":

    bounds = Bounds(
        lb=np.array([-5.0, -5.0]),
        ub=np.array([5.0, 5.0])
    )

    starting_points = [
        np.array([0.0, 0.0]),
        np.array([4.0, 4.0]),
        np.array([-4.0, 4.0]),
        np.array([-4.0, -4.0]),
        np.array([4.0, -4.0]),
    ]

    print("Library NLP solver results for box-constrained Himmelblau problem")
    print("=" * 70)

    for x0 in starting_points:
        res = minimize(
            fun=himmelblau,
            x0=x0,
            jac=grad_himmelblau,
            hess=hess_himmelblau,
            bounds=bounds,
            method="trust-constr",
            options={
                "gtol": 1e-10,
                "xtol": 1e-10,
                "maxiter": 1000,
                "verbose": 0,
            }
        )

        print(f"Initial point:   {x0}")
        print(f"Solution:        {res.x}")
        print(f"Objective value: {res.fun:.10e}")
        print(f"Iterations:      {res.niter}")
        print(f"Success:         {res.success}")
        print(f"Message:         {res.message}")
        print("-" * 70)