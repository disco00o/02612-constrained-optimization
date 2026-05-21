import numpy as np


def kkt_residuals(x, grad_f, g, jac_g, gl, gu, xl, xu,
                  lambda_l, lambda_u, mu_l, mu_u):
    """
    Compute KKT residuals for the NLP:

        min f(x)
        s.t. gl <= g(x) <= gu
             xl <= x <= xu
    """

    x = np.asarray(x)

    # Stationarity residual
    stationarity = (
        grad_f(x)
        + jac_g(x).T @ (lambda_u - lambda_l)
        + mu_u
        - mu_l
    )

    # Primal feasibility residuals
    nonlinear_lower_violation = np.maximum(gl - g(x), 0)
    nonlinear_upper_violation = np.maximum(g(x) - gu, 0)

    box_lower_violation = np.maximum(xl - x, 0)
    box_upper_violation = np.maximum(x - xu, 0)

    # Complementarity residuals
    comp_nonlinear_lower = lambda_l * (gl - g(x))
    comp_nonlinear_upper = lambda_u * (g(x) - gu)

    comp_box_lower = mu_l * (xl - x)
    comp_box_upper = mu_u * (x - xu)

    return {
        "stationarity": stationarity,
        "nonlinear_lower_violation": nonlinear_lower_violation,
        "nonlinear_upper_violation": nonlinear_upper_violation,
        "box_lower_violation": box_lower_violation,
        "box_upper_violation": box_upper_violation,
        "comp_nonlinear_lower": comp_nonlinear_lower,
        "comp_nonlinear_upper": comp_nonlinear_upper,
        "comp_box_lower": comp_box_lower,
        "comp_box_upper": comp_box_upper,
    }

if __name__ == "__main__":

    def grad_f(x):
        return np.array([2*x[0], 2*x[1]])

    def g(x):
        return np.array([x[0] + x[1]])

    def jac_g(x):
        return np.array([[1.0, 1.0]])

    x = np.array([1.0, 2.0])

    gl = np.array([0.0])
    gu = np.array([10.0])

    xl = np.array([-5.0, -5.0])
    xu = np.array([5.0, 5.0])

    lambda_l = np.array([0.0])
    lambda_u = np.array([1.0])

    mu_l = np.array([0.0, 0.0])
    mu_u = np.array([0.0, 0.0])

    residuals = kkt_residuals(
        x,
        grad_f,
        g,
        jac_g,
        gl,
        gu,
        xl,
        xu,
        lambda_l,
        lambda_u,
        mu_l,
        mu_u
    )

    print(residuals)