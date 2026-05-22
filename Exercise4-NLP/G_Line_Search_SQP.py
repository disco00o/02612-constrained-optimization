import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds, LinearConstraint


FIGURE_DIR = "Exercise4-NLP/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


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


def g_constraint(x):
    return np.array([
        x[0] + x[1]
    ])


def jac_g_constraint(x):
    return np.array([
        [1.0, 1.0]
    ])


def constraint_violation(g_fun, x, gl, gu):

    if g_fun is None:
        return 0.0

    gx = g_fun(x)

    lower_violation = np.maximum(gl - gx, 0.0)
    upper_violation = np.maximum(gx - gu, 0.0)

    return np.sum(lower_violation + upper_violation)


def merit_function(f_fun, g_fun, x, gl, gu, penalty):

    return (
        f_fun(x)
        + penalty * constraint_violation(g_fun, x, gl, gu)
    )


def damped_bfgs_update(Bk, s, y):

    Bs = Bk @ s

    sBs = s @ Bs
    sy = s @ y

    if sBs <= 1e-14:
        return Bk

    if sy < 0.2 * sBs:

        theta = (0.8 * sBs) / (sBs - sy)

        y = theta * y + (1.0 - theta) * Bs

    sy = s @ y

    if sy <= 1e-14:
        return Bk

    Bnew = (
        Bk
        - np.outer(Bs, Bs) / sBs
        + np.outer(y, y) / sy
    )

    return Bnew


def solve_qp_subproblem(
    xk,
    Hk,
    grad_fk,
    xl,
    xu,
    g_fun=None,
    jac_g_fun=None,
    gl=None,
    gu=None
):

    n = len(xk)

    def qp_obj(p):
        return 0.5 * p @ Hk @ p + grad_fk @ p

    def qp_grad(p):
        return Hk @ p + grad_fk

    bounds = Bounds(
        xl - xk,
        xu - xk
    )

    constraints = []

    if g_fun is not None and jac_g_fun is not None:

        gk = g_fun(xk)
        Jgk = jac_g_fun(xk)

        constraints.append(
            LinearConstraint(
                Jgk,
                gl - gk,
                gu - gk
            )
        )

    res = minimize(
        fun=qp_obj,
        x0=np.zeros(n),
        jac=qp_grad,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={
            "ftol": 1e-12,
            "maxiter": 300,
            "disp": False
        }
    )

    return res.x


def line_search_sqp(
    f_fun,
    grad_f_fun,
    hess_f_fun,
    x0,
    xl,
    xu,
    g_fun=None,
    jac_g_fun=None,
    gl=None,
    gu=None,
    max_iter=100,
    tol=1e-8,
    penalty=100.0,
    hessian_mode="exact"
):

    xk = np.asarray(x0, dtype=float)

    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    n = len(xk)

    Bk = np.eye(n)

    history = {
        "x": [],
        "f": [],
        "merit": [],
        "grad_norm": [],
        "alpha": [],
        "constraint_violation": []
    }

    for k in range(max_iter):

        fk = f_fun(xk)

        grad_fk = grad_f_fun(xk)

        violation = constraint_violation(
            g_fun,
            xk,
            gl,
            gu
        )

        history["x"].append(xk.copy())

        history["f"].append(fk)

        history["merit"].append(
            merit_function(
                f_fun,
                g_fun,
                xk,
                gl,
                gu,
                penalty
            )
        )

        history["grad_norm"].append(
            np.linalg.norm(grad_fk)
        )

        history["constraint_violation"].append(
            violation
        )

        if (
            np.linalg.norm(grad_fk) < tol
            and violation < tol
        ):
            break

        if hessian_mode == "exact":
            Hk = hess_f_fun(xk)

        elif hessian_mode == "bfgs":
            Hk = Bk

        else:
            raise ValueError(
                "hessian_mode must be 'exact' or 'bfgs'."
            )

        eig_min = np.min(
            np.linalg.eigvalsh(Hk)
        )

        if eig_min <= 1e-8:

            Hk = (
                Hk
                + (abs(eig_min) + 1e-4)
                * np.eye(n)
            )

        pk = solve_qp_subproblem(
            xk=xk,
            Hk=Hk,
            grad_fk=grad_fk,
            xl=xl,
            xu=xu,
            g_fun=g_fun,
            jac_g_fun=jac_g_fun,
            gl=gl,
            gu=gu
        )

        alpha = 1.0
        rho = 0.5
        c1 = 1e-4

        current_merit = merit_function(
            f_fun,
            g_fun,
            xk,
            gl,
            gu,
            penalty
        )

        while alpha > 1e-12:

            x_trial = xk + alpha * pk

            x_trial = np.minimum(
                np.maximum(x_trial, xl),
                xu
            )

            trial_merit = merit_function(
                f_fun,
                g_fun,
                x_trial,
                gl,
                gu,
                penalty
            )

            if (
                trial_merit
                <= current_merit
                - c1 * alpha * np.linalg.norm(pk)**2
            ):
                break

            alpha *= rho

        x_next = x_trial

        if hessian_mode == "bfgs":

            s = x_next - xk

            y = (
                grad_f_fun(x_next)
                - grad_fk
            )

            Bk = damped_bfgs_update(
                Bk,
                s,
                y
            )

        history["alpha"].append(alpha)

        if np.linalg.norm(x_next - xk) < tol:
            xk = x_next
            break

        xk = x_next

    return xk, f_fun(xk), history


def plot_convergence(history, filename, title):

    plt.figure(figsize=(7, 5))

    plt.semilogy(
        history["f"],
        marker="o"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    xl = np.array([-5.0, -5.0])
    xu = np.array([5.0, 5.0])

    gl = np.array([-5.0])
    gu = np.array([5.0])

    starting_points = [
        np.array([0.0, 0.0]),
        np.array([4.0, 4.0]),
        np.array([-4.0, 4.0]),
        np.array([-4.0, -4.0]),
        np.array([4.0, -4.0])
    ]

    for mode in ["exact", "bfgs"]:

        print(
            f"\nLine-search SQP using {mode} Hessian"
        )

        print("=" * 70)

        for x0 in starting_points:

            x_star, f_star, history = line_search_sqp(
                f_fun=himmelblau,
                grad_f_fun=grad_himmelblau,
                hess_f_fun=hess_himmelblau,
                x0=x0,
                xl=xl,
                xu=xu,
                g_fun=g_constraint,
                jac_g_fun=jac_g_constraint,
                gl=gl,
                gu=gu,
                hessian_mode=mode
            )

            print(f"Initial point:          {x0}")
            print(f"Solution:               {x_star}")
            print(f"Objective value:        {f_star:.10e}")
            print(f"Constraint value g(x):  {g_constraint(x_star)}")
            print(f"Iterations:             {len(history['f'])}")

            print("-" * 70)

        plot_convergence(
            history,
            filename=f"line_search_sqp_{mode}_convergence.png",
            title=f"Line-search SQP convergence ({mode})"
        )