import numpy as np
from scipy.optimize import Bounds, LinearConstraint


def exact_hessian_lagrangian(
    x,
    hess_f,
    hess_g,
    lambda_k
):
    """
    Compute the exact Hessian of the Lagrangian:

        L(x, lambda)
        = f(x) + sum_i lambda_i g_i(x)

    Parameters
    ----------
    x : ndarray
        Current iterate

    hess_f : callable
        Hessian of objective function

    hess_g : list of callables
        Hessians of constraint functions

    lambda_k : ndarray
        Current Lagrange multipliers

    Returns
    -------
    H : ndarray
        Exact Hessian of the Lagrangian
    """

    H = hess_f(x).copy()

    for i in range(len(lambda_k)):
        H += lambda_k[i] * hess_g[i](x)

    return H


def damped_bfgs_update(
    Bk,
    s,
    y
):
    """
    Powell damped BFGS Hessian update.

    Parameters
    ----------
    Bk : ndarray
        Current Hessian approximation

    s : ndarray
        Step vector:
            s = x_{k+1} - x_k

    y : ndarray
        Gradient difference:
            y = grad_{k+1} - grad_k

    Returns
    -------
    Bnew : ndarray
        Updated Hessian approximation
    """

    Bs = Bk @ s

    sBs = s @ Bs
    sy = s @ y

    # Powell damping
    if sy < 0.2 * sBs:

        theta = (
            0.8 * sBs
            / (sBs - sy)
        )

        y = (
            theta * y
            + (1 - theta) * Bs
        )

    sy = s @ y

    # Safeguard
    if sy <= 1e-12:
        return Bk

    Bnew = (
        Bk
        - np.outer(Bs, Bs) / sBs
        + np.outer(y, y) / sy
    )

    return Bnew


# ============================================================
# Build SQP Quadratic Subproblem
# ============================================================

def build_sqp_qp_subproblem(
    xk,
    grad_f,
    g,
    jac_g,
    gl,
    gu,
    xl,
    xu,
    Hk
):
    """
    Construct the SQP quadratic programming subproblem:

        min_p
            0.5 p^T Hk p + grad_f(xk)^T p

        s.t.
            gl <= g(xk) + Jg(xk)p <= gu
            xl <= xk + p <= xu

    Parameters
    ----------
    xk : ndarray
        Current iterate

    grad_f : callable
        Gradient of objective function

    g : callable
        Constraint function

    jac_g : callable
        Jacobian of constraints

    gl, gu : ndarray
        Lower and upper nonlinear constraint bounds

    xl, xu : ndarray
        Variable lower and upper bounds

    Hk : ndarray
        Hessian approximation

    Returns
    -------
    qp_data : dict
        Dictionary containing all QP quantities
    """

    # ========================================================
    # Quadratic objective
    # ========================================================

    q = grad_f(xk)

    # ========================================================
    # Linearized nonlinear constraints
    # ========================================================

    if g is not None and jac_g is not None:

        gk = g(xk)

        Jg = jac_g(xk)

        nonlinear_constraint = LinearConstraint(
            Jg,
            gl - gk,
            gu - gk
        )

    else:

        nonlinear_constraint = None

    # ========================================================
    # Box constraints on step p
    # ========================================================

    bounds = Bounds(
        xl - xk,
        xu - xk
    )

    qp_data = {
        "H": Hk,
        "q": q,
        "bounds": bounds,
        "nonlinear_constraint": nonlinear_constraint
    }

    return qp_data


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example objective
    # --------------------------------------------------------

    def f(x):
        return x[0]**2 + x[1]**2

    def grad_f(x):
        return np.array([
            2 * x[0],
            2 * x[1]
        ])

    def hess_f(x):
        return np.array([
            [2.0, 0.0],
            [0.0, 2.0]
        ])

    # --------------------------------------------------------
    # Example nonlinear constraint
    #
    #   g(x) = x1 + x2
    #
    # --------------------------------------------------------

    def g(x):
        return np.array([
            x[0] + x[1]
        ])

    def jac_g(x):
        return np.array([
            [1.0, 1.0]
        ])

    def hess_g1(x):
        return np.zeros((2, 2))

    hess_g = [hess_g1]

    # --------------------------------------------------------
    # Current iterate
    # --------------------------------------------------------

    xk = np.array([1.0, 2.0])

    lambda_k = np.array([1.0])

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    gl = np.array([0.0])
    gu = np.array([10.0])

    xl = np.array([-5.0, -5.0])
    xu = np.array([5.0, 5.0])

    # --------------------------------------------------------
    # Exact Hessian
    # --------------------------------------------------------

    Hk = exact_hessian_lagrangian(
        x=xk,
        hess_f=hess_f,
        hess_g=hess_g,
        lambda_k=lambda_k
    )

    print("Exact Hessian:")
    print(Hk)

    # --------------------------------------------------------
    # Build SQP QP subproblem
    # --------------------------------------------------------

    qp_data = build_sqp_qp_subproblem(
        xk=xk,
        grad_f=grad_f,
        g=g,
        jac_g=jac_g,
        gl=gl,
        gu=gu,
        xl=xl,
        xu=xu,
        Hk=Hk
    )

    print("\nQP Subproblem Data:")
    print("H:")
    print(qp_data["H"])

    print("\nq:")
    print(qp_data["q"])

    print("\nBounds:")
    print(qp_data["bounds"])

    print("\nLinearized Nonlinear Constraint:")
    print(qp_data["nonlinear_constraint"])