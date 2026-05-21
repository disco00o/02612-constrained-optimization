import numpy as np


def lagrangian(x, f, g, gl, gu, xl, xu,
               lambda_l, lambda_u, mu_l, mu_u):
    """
    Evaluate the Lagrangian for the NLP:

        min f(x)
        s.t. gl <= g(x) <= gu
             xl <= x <= xu
    """

    x = np.asarray(x)
    gl = np.asarray(gl)
    gu = np.asarray(gu)
    xl = np.asarray(xl)
    xu = np.asarray(xu)

    lambda_l = np.asarray(lambda_l)
    lambda_u = np.asarray(lambda_u)
    mu_l = np.asarray(mu_l)
    mu_u = np.asarray(mu_u)

    return (
        f(x)
        + lambda_u @ (g(x) - gu)
        + lambda_l @ (gl - g(x))
        + mu_u @ (x - xu)
        + mu_l @ (xl - x)
    )

if __name__ == "__main__":

    def f(x):
        return x[0]**2 + x[1]**2

    def g(x):
        return np.array([x[0] + x[1]])

    x = np.array([1.0, 2.0])

    gl = np.array([0.0])
    gu = np.array([10.0])

    xl = np.array([-5.0, -5.0])
    xu = np.array([5.0, 5.0])

    lambda_l = np.array([0.0])
    lambda_u = np.array([1.0])

    mu_l = np.array([0.0, 0.0])
    mu_u = np.array([0.0, 0.0])

    L = lagrangian(
        x,
        f,
        g,
        gl,
        gu,
        xl,
        xu,
        lambda_l,
        lambda_u,
        mu_l,
        mu_u
    )

    print("Lagrangian value:", L)