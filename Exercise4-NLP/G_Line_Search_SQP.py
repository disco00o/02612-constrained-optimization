import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds


FIGURE_DIR = "Exercise4-NLP/figures"

os.makedirs(FIGURE_DIR, exist_ok=True)


def himmelblau(x):
    x1, x2 = x

    return (
        (x1**2 + x2 - 11)**2
        + (x1 + x2**2 - 7)**2
    )


def grad_himmelblau(x):
    x1, x2 = x

    return np.array([
        4 * x1 * (x1**2 + x2 - 11)
        + 2 * (x1 + x2**2 - 7),

        2 * (x1**2 + x2 - 11)
        + 4 * x2 * (x1 + x2**2 - 7)
    ])


def hess_himmelblau(x):
    x1, x2 = x

    return np.array([
        [12 * x1**2 + 4 * x2 - 42,
         4 * x1 + 4 * x2],

        [4 * x1 + 4 * x2,
         4 * x1 + 12 * x2**2 - 26]
    ])


def damped_bfgs_update(Bk, s, y):

    Bs = Bk @ s

    sBs = s @ Bs
    sy = s @ y

    if sy < 0.2 * sBs:

        theta = (0.8 * sBs) / (sBs - sy)

        y = theta * y + (1 - theta) * Bs

    sy = s @ y

    if sy <= 1e-12:
        return Bk

    return (
        Bk
        - np.outer(Bs, Bs) / sBs
        + np.outer(y, y) / sy
    )


def solve_qp_subproblem(
    xk,
    Hk,
    grad_fk,
    xl,
    xu
):

    n = len(xk)

    def qp_obj(p):
        return (
            0.5 * p @ Hk @ p
            + grad_fk @ p
        )

    def qp_grad(p):
        return Hk @ p + grad_fk

    bounds = Bounds(
        xl - xk,
        xu - xk
    )

    res = minimize(
        fun=qp_obj,
        x0=np.zeros(n),
        jac=qp_grad,
        bounds=bounds,
        method="SLSQP",
        options={
            "ftol": 1e-12,
            "maxiter": 200
        }
    )

    return res.x


def line_search_sqp(
    x0,
    xl,
    xu,
    hessian_mode="exact",
    max_iter=100,
    tol=1e-8
):

    xk = np.asarray(x0, dtype=float)

    n = len(xk)

    Bk = np.eye(n)

    history = {
        "f": []
    }

    for k in range(max_iter):

        fk = himmelblau(xk)

        grad_fk = grad_himmelblau(xk)

        history["f"].append(fk)

        if np.linalg.norm(grad_fk) < tol:
            break

        if hessian_mode == "exact":
            Hk = hess_himmelblau(xk)
        else:
            Hk = Bk

        eig_min = np.min(np.linalg.eigvalsh(Hk))

        if eig_min <= 1e-8:
            Hk += (
                abs(eig_min) + 1e-4
            ) * np.eye(n)

        pk = solve_qp_subproblem(
            xk=xk,
            Hk=Hk,
            grad_fk=grad_fk,
            xl=xl,
            xu=xu
        )

        alpha = 1.0

        rho = 0.5

        c1 = 1e-4

        while alpha > 1e-12:

            x_trial = xk + alpha * pk

            x_trial = np.minimum(
                np.maximum(x_trial, xl),
                xu
            )

            if (
                himmelblau(x_trial)
                <= fk + c1 * alpha * grad_fk @ pk
            ):
                break

            alpha *= rho

        x_next = x_trial

        if hessian_mode == "bfgs":

            s = x_next - xk

            y = (
                grad_himmelblau(x_next)
                - grad_fk
            )

            Bk = damped_bfgs_update(
                Bk,
                s,
                y
            )

        if np.linalg.norm(x_next - xk) < tol:
            xk = x_next
            break

        xk = x_next

    return (
        xk,
        himmelblau(xk),
        history
    )


def plot_convergence(
    history,
    filename
):

    plt.figure(figsize=(7, 5))

    plt.semilogy(
        history["f"],
        marker="o"
    )

    plt.xlabel("Iteration")

    plt.ylabel("Objective value")

    plt.title(
        "Line-search SQP convergence"
    )

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

        print("=" * 60)

        for x0 in starting_points:

            x_star, f_star, history = line_search_sqp(
                x0=x0,
                xl=xl,
                xu=xu,
                hessian_mode=mode
            )

            print(f"Initial point:   {x0}")

            print(f"Solution:        {x_star}")

            print(
                f"Objective value: {f_star:.10e}"
            )

            print(
                f"Iterations:      {len(history['f'])}"
            )

            print("-" * 60)

        plot_convergence(
            history,
            filename=f"line_search_sqp_{mode}_convergence.png"
        )