import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds


FIGURE_DIR = "Exercise4-NLP/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# Test Problems
# ============================================================

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


def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


# ============================================================
# QP Solvers Used Inside SQP
# ============================================================

def solve_qp_subproblem(xk, Hk, grad_fk, xl, xu):
    n = len(xk)

    def qp_obj(p):
        return 0.5 * p @ Hk @ p + grad_fk @ p

    def qp_grad(p):
        return Hk @ p + grad_fk

    bounds = Bounds(xl - xk, xu - xk)

    res = minimize(
        fun=qp_obj,
        x0=np.zeros(n),
        jac=qp_grad,
        bounds=bounds,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 200}
    )

    return res.x


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
        options={"ftol": 1e-12, "maxiter": 200}
    )

    return res.x


# ============================================================
# Simplified SQP Implementations
# ============================================================

def line_search_sqp(x0, xl, xu, max_iter=100, tol=1e-8):
    xk = np.asarray(x0, dtype=float)

    history = {"f": []}

    for _ in range(max_iter):
        fk = himmelblau(xk)
        grad_fk = grad_himmelblau(xk)
        Hk = hess_himmelblau(xk)

        history["f"].append(fk)

        if np.linalg.norm(grad_fk) < tol:
            break

        eig_min = np.min(np.linalg.eigvalsh(Hk))
        if eig_min <= 1e-8:
            Hk = Hk + (abs(eig_min) + 1e-4) * np.eye(len(xk))

        pk = solve_qp_subproblem(xk, Hk, grad_fk, xl, xu)

        alpha = 1.0
        rho = 0.5
        c1 = 1e-4

        while alpha > 1e-12:
            x_trial = np.minimum(np.maximum(xk + alpha * pk, xl), xu)

            if himmelblau(x_trial) <= fk + c1 * alpha * grad_fk @ pk:
                break

            alpha *= rho

        x_next = x_trial

        if np.linalg.norm(x_next - xk) < tol:
            xk = x_next
            break

        xk = x_next

    return xk, himmelblau(xk), history


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

    history = {"f": [], "delta": []}

    for _ in range(max_iter):
        fk = himmelblau(xk)
        grad_fk = grad_himmelblau(xk)
        Hk = hess_himmelblau(xk)

        history["f"].append(fk)
        history["delta"].append(delta)

        if np.linalg.norm(grad_fk) < tol:
            break

        eig_min = np.min(np.linalg.eigvalsh(Hk))
        if eig_min <= 1e-8:
            Hk = Hk + (abs(eig_min) + 1e-4) * np.eye(len(xk))

        pk = solve_trust_region_qp(xk, Hk, grad_fk, xl, xu, delta)

        predicted = -(grad_fk @ pk + 0.5 * pk @ Hk @ pk)
        x_trial = np.minimum(np.maximum(xk + pk, xl), xu)
        actual = fk - himmelblau(x_trial)

        rho = 0.0 if predicted <= 1e-12 else actual / predicted

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


# ============================================================
# Solver Wrappers
# ============================================================

def solve_with_library(problem, x0, bounds):
    start = time.perf_counter()

    res = minimize(
        fun=problem,
        x0=x0,
        method="trust-constr",
        bounds=Bounds(
            np.array([b[0] for b in bounds]),
            np.array([b[1] for b in bounds])
        ),
        options={"maxiter": 1000}
    )

    cpu_time = time.perf_counter() - start

    return {
        "solver": "Library Solver",
        "objective": res.fun,
        "iterations": res.niter,
        "cpu_time": cpu_time,
        "success": res.success
    }


def run_line_search_sqp(problem, x0, bounds):
    start = time.perf_counter()

    xl = np.array([b[0] for b in bounds])
    xu = np.array([b[1] for b in bounds])

    x_star, f_star, history = line_search_sqp(
        x0=x0,
        xl=xl,
        xu=xu
    )

    cpu_time = time.perf_counter() - start

    return {
        "solver": "Line-Search SQP",
        "objective": f_star,
        "iterations": len(history["f"]),
        "cpu_time": cpu_time,
        "success": True
    }


def run_trust_region_sqp(problem, x0, bounds):
    start = time.perf_counter()

    xl = np.array([b[0] for b in bounds])
    xu = np.array([b[1] for b in bounds])

    x_star, f_star, history = trust_region_sqp(
        x0=x0,
        xl=xl,
        xu=xu
    )

    cpu_time = time.perf_counter() - start

    return {
        "solver": "Trust-Region SQP",
        "objective": f_star,
        "iterations": len(history["f"]),
        "cpu_time": cpu_time,
        "success": True
    }


# ============================================================
# Main Experiment
# ============================================================

if __name__ == "__main__":

    problems = {
        "Himmelblau": {
            "func": himmelblau,
            "x0": np.array([0.0, 0.0]),
            "bounds": [(-5, 5), (-5, 5)]
        },

        "Rosenbrock": {
            "func": rosenbrock,
            "x0": np.array([-1.2, 1.0]),
            "bounds": [(-5, 5), (-5, 5)]
        }
    }

    solvers = [
        solve_with_library,
        run_line_search_sqp,
        run_trust_region_sqp
    ]

    all_results = []

    for problem_name, data in problems.items():

        print(f"\nRunning problem: {problem_name}")
        print("=" * 60)

        for solver in solvers:

            result = solver(
                data["func"],
                data["x0"],
                data["bounds"]
            )

            result["problem"] = problem_name
            all_results.append(result)

            print(result)

    df = pd.DataFrame(all_results)

    print("\nSummary Table")
    print(df)

    df.to_csv(
        f"{FIGURE_DIR}/nlp_solver_comparison.csv",
        index=False
    )

    # ========================================================
    # Improved CPU Time Plot
    # ========================================================

    plt.figure(figsize=(10, 5))

    problem_names = df["problem"].unique()
    solver_names = df["solver"].unique()

    x = np.arange(len(solver_names))
    width = 0.35

    for i, problem_name in enumerate(problem_names):

        subset = df[df["problem"] == problem_name]

        cpu_times = [
            subset[subset["solver"] == solver]["cpu_time"].values[0]
            for solver in solver_names
        ]

        plt.bar(
            x + i * width,
            cpu_times,
            width=width,
            label=problem_name
        )

    plt.xticks(
        x + width / 2,
        solver_names
    )

    plt.ylabel("CPU Time [s]")
    plt.title("NLP Solver CPU Time Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/nlp_solver_cpu_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()