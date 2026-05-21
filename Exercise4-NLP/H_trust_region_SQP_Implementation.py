import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds


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


def solve_trust_region_qp(xk, Hk, grad_fk, xl, xu, delta):
    n = len(xk)

    def qp_obj(p):
        return 0.5 * p @ Hk @ p + grad_fk @ p

    def qp_grad(p):
        return Hk @ p + grad_fk

    bounds = Bounds(xl - xk, xu - xk)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda p: delta**2 - p @ p,
            "jac": lambda p: -2 * p
        }
    ]

    res = minimize(
        fun=qp_obj,
        x0=np.zeros(n),
        jac=qp_grad,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={
            "ftol": 1e-12,
            "maxiter": 200
        }
    )

    return res.x


def trust_region_sqp(
    x0,
    xl,
    xu,
    max_iter=100,
    tol=1e-8,
    delta0=1.0,
    delta_max=10.0,
    eta=0.1
):
    xk = np.asarray(x0, dtype=float)
    delta = delta0

    history = {
        "x": [],
        "f": [],
        "grad_norm": [],
        "delta": [],
        "rho": []
    }

    for k in range(max_iter):
        fk = himmelblau(xk)
        grad_fk = grad_himmelblau(xk)
        Hk = hess_himmelblau(xk)

        history["x"].append(xk.copy())
        history["f"].append(fk)
        history["grad_norm"].append(np.linalg.norm(grad_fk))
        history["delta"].append(delta)

        if np.linalg.norm(grad_fk) < tol:
            break

        eig_min = np.min(np.linalg.eigvalsh(Hk))

        if eig_min <= 1e-8:
            Hk = Hk + (abs(eig_min) + 1e-4) * np.eye(len(xk))

        pk = solve_trust_region_qp(
            xk=xk,
            Hk=Hk,
            grad_fk=grad_fk,
            xl=xl,
            xu=xu,
            delta=delta
        )

        predicted_reduction = -(grad_fk @ pk + 0.5 * pk @ Hk @ pk)

        x_trial = np.minimum(
            np.maximum(xk + pk, xl),
            xu
        )

        actual_reduction = fk - himmelblau(x_trial)

        if predicted_reduction <= 1e-12:
            rho = 0.0
        else:
            rho = actual_reduction / predicted_reduction

        history["rho"].append(rho)

        if rho < 0.25:
            delta *= 0.25

        elif rho > 0.75 and np.linalg.norm(pk) >= 0.9 * delta:
            delta = min(2.0 * delta, delta_max)

        if rho > eta:
            x_next = x_trial
        else:
            x_next = xk

        if (
            np.linalg.norm(x_next - xk) < tol
            or np.linalg.norm(grad_himmelblau(x_next)) < tol
        ):
            xk = x_next
            break

        xk = x_next

    return xk, himmelblau(xk), history


def plot_convergence(history, filename):
    plt.figure(figsize=(7, 5))

    plt.semilogy(
        history["f"],
        marker="o"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("Trust-region SQP convergence")
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

    print("\nTrust-region SQP on box-constrained Himmelblau")
    print("=" * 70)

    for x0 in starting_points:
        x_star, f_star, history = trust_region_sqp(
            x0=x0,
            xl=xl,
            xu=xu,
            delta0=1.0
        )

        print(f"Initial point:   {x0}")
        print(f"Solution:        {x_star}")
        print(f"Objective value: {f_star:.10e}")
        print(f"Iterations:      {len(history['f'])}")
        print(f"Final radius:    {history['delta'][-1]:.4e}")
        print("-" * 70)

    plot_convergence(
        history,
        filename="trust_region_sqp_convergence.png"
    )