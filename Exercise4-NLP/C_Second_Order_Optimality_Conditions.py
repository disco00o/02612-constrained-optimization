import numpy as np


def hessian_lagrangian(x, hess_f, hess_g, lambda_l, lambda_u):
    """
    Compute Hessian of the Lagrangian:

        ∇²_xx L = ∇²f(x) + sum_i (lambda_u_i - lambda_l_i) ∇²g_i(x)

    hess_g(x) should return a list/array of Hessian matrices for each g_i.
    """

    H = hess_f(x).copy()
    G_hessians = hess_g(x)

    for i in range(len(lambda_u)):
        H += (lambda_u[i] - lambda_l[i]) * G_hessians[i]

    return H


def check_second_order_sufficient(H_L, directions, tol=1e-10):
    """
    Numerically check d^T H_L d > 0 for a collection of feasible directions.

    This does not prove the condition globally, but is useful for testing.
    """

    values = []

    for d in directions:
        d = np.asarray(d)
        value = d.T @ H_L @ d
        values.append(value)

        if value <= tol:
            return False, values

    return True, values

if __name__ == "__main__":

    def hess_f(x):
        return np.array([
            [2.0, 0.0],
            [0.0, 2.0]
        ])

    def hess_g(x):
        return [
            np.array([
                [1.0, 0.0],
                [0.0, 1.0]
            ])
        ]

    x = np.array([1.0, 2.0])

    lambda_l = np.array([0.0])
    lambda_u = np.array([1.0])

    H = hessian_lagrangian(
        x,
        hess_f,
        hess_g,
        lambda_l,
        lambda_u
    )

    print("Hessian of Lagrangian:")
    print(H)