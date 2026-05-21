import numpy as np


def exact_hessian_lagrangian(x, hess_f, hess_g, lambda_k):
    """
    Exact Hessian of the Lagrangian.
    """

    H = hess_f(x).copy()

    for i in range(len(lambda_k)):
        H += lambda_k[i] * hess_g[i](x)

    return H


def damped_bfgs_update(Bk, s, y):
    """
    Damped BFGS Hessian update.
    """

    Bs = Bk @ s
    sBs = s.T @ Bs
    sy = s.T @ y

    if sy < 0.2 * sBs:
        theta = (0.8 * sBs) / (sBs - sy)
        y = theta * y + (1 - theta) * Bs

    rho = 1.0 / (s.T @ y)
    I = np.eye(len(s))

    Bnew = (
        Bk
        - np.outer(Bs, Bs) / sBs
        + np.outer(y, y) * rho
    )

    return Bnew


def build_sqp_qp_subproblem(xk, grad_f, g, jac_g, gl, gu, xl, xu, Hk):
    """
    Construct the SQP QP subproblem.
    """

    q = grad_f(xk)
    A = jac_g(xk)

    bl = gl - g(xk)
    bu = gu - g(xk)

    pl = xl - xk
    pu = xu - xk

    return Hk, q, A, bl, bu, pl, pu


def build_sqp_qp_with_hessian_choice(
    xk,
    grad_f,
    g,
    jac_g,
    gl,
    gu,
    xl,
    xu,
    hessian_mode,
    hess_f=None,
    hess_g=None,
    lambda_k=None,
    Bk=None,
):
    """
    Build SQP QP subproblem using either exact Hessian or BFGS approximation.
    """

    if hessian_mode == "exact":
        Hk = exact_hessian_lagrangian(xk, hess_f, hess_g, lambda_k)

    elif hessian_mode == "bfgs":
        Hk = Bk

    else:
        raise ValueError("hessian_mode must be either 'exact' or 'bfgs'.")

    return build_sqp_qp_subproblem(
        xk=xk,
        grad_f=grad_f,
        g=g,
        jac_g=jac_g,
        gl=gl,
        gu=gu,
        xl=xl,
        xu=xu,
        Hk=Hk,
    )