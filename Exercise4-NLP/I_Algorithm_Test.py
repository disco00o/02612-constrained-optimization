import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds, LinearConstraint


FIGURE_DIR = "Exercise4-NLP/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

BIG = 1e20


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


def c_himmelblau(x):
    x1, x2 = x
    return np.array([
        (x1 + 2)**2 - x2,
        -4 * x1 + 10 * x2
    ])


def jac_c_himmelblau(x):
    x1, x2 = x
    return np.array([
        [2 * (x1 + 2), -1.0],
        [-4.0, 10.0]
    ])


def constraint_violation(g_fun, x, gl, gu):
    if g_fun is None:
        return 0.0

    gx = g_fun(x)
    lower_violation = np.maximum(gl - gx, 0.0)
    upper_violation = np.maximum(gx - gu, 0.0)

    return np.sum(lower_violation + upper_violation)


def merit_function(f_fun, g_fun, x, gl, gu, penalty):
    return f_fun(x) + penalty * constraint_violation(g_fun, x, gl, gu)


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

    bounds = Bounds(xl - xk, xu - xk)

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


def solve_trust_region_qp(
    xk,
    Hk,
    grad_fk,
    xl,
    xu,
    delta,
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

    bounds = Bounds(xl - xk, xu - xk)

    constraints = [
        {
            "type": "ineq",
            "fun": lambda p: delta**2 - p @ p,
            "jac": lambda p: -2 * p
        }
    ]

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
    penalty=100.0
):
    xk = np.asarray(x0, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    history = {
        "f": [],
        "constraint_violation": []
    }

    for _ in range(max_iter):
        fk = f_fun(xk)
        grad_fk = grad_f_fun(xk)
        Hk = hess_f_fun(xk)

        violation = constraint_violation(g_fun, xk, gl, gu)

        history["f"].append(fk)
        history["constraint_violation"].append(violation)

        if np.linalg.norm(grad_fk) < tol and violation < tol:
            break

        eig_min = np.min(np.linalg.eigvalsh(Hk))

        if eig_min <= 1e-8:
            Hk = Hk + (abs(eig_min) + 1e-4) * np.eye(len(xk))

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
            x_trial = np.minimum(np.maximum(x_trial, xl), xu)

            trial_merit = merit_function(
                f_fun,
                g_fun,
                x_trial,
                gl,
                gu,
                penalty
            )

            if trial_merit <= current_merit - c1 * alpha * np.linalg.norm(pk)**2:
                break

            alpha *= rho

        x_next = x_trial

        if np.linalg.norm(x_next - xk) < tol:
            xk = x_next
            break

        xk = x_next

    return xk, f_fun(xk), history


def trust_region_sqp(
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
    delta0=1.0,
    delta_max=10.0,
    eta=0.1
):
    xk = np.asarray(x0, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    delta = delta0

    history = {
        "f": [],
        "constraint_violation": [],
        "delta": []
    }

    for _ in range(max_iter):
        fk = f_fun(xk)
        grad_fk = grad_f_fun(xk)
        Hk = hess_f_fun(xk)

        violation = constraint_violation(g_fun, xk, gl, gu)

        history["f"].append(fk)
        history["constraint_violation"].append(violation)
        history["delta"].append(delta)

        if np.linalg.norm(grad_fk) < tol and violation < tol:
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
            delta=delta,
            g_fun=g_fun,
            jac_g_fun=jac_g_fun,
            gl=gl,
            gu=gu
        )

        predicted = -(grad_fk @ pk + 0.5 * pk @ Hk @ pk)

        x_trial = np.minimum(np.maximum(xk + pk, xl), xu)

        actual = fk - f_fun(x_trial)

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
            or np.linalg.norm(grad_f_fun(x_next)) < tol
        ):
            xk = x_next
            break

        xk = x_next

    return xk, f_fun(xk), history


def solve_with_library(problem):
    start = time.perf_counter()

    scipy_constraints = []

    if problem["g"] is not None:
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, problem=problem: problem["g"](x) - problem["gl"],
                "jac": lambda x, problem=problem: problem["jac_g"](x)
            }
        )

        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, problem=problem: problem["gu"] - problem["g"](x),
                "jac": lambda x, problem=problem: -problem["jac_g"](x)
            }
        )

    res = minimize(
        fun=problem["f"],
        x0=problem["x0"],
        jac=problem["grad_f"],
        hess=problem["hess_f"],
        method="trust-constr",
        bounds=Bounds(problem["xl"], problem["xu"]),
        constraints=scipy_constraints,
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


def run_line_search_sqp(problem):
    start = time.perf_counter()

    x_star, f_star, history = line_search_sqp(
        f_fun=problem["f"],
        grad_f_fun=problem["grad_f"],
        hess_f_fun=problem["hess_f"],
        x0=problem["x0"],
        xl=problem["xl"],
        xu=problem["xu"],
        g_fun=problem["g"],
        jac_g_fun=problem["jac_g"],
        gl=problem["gl"],
        gu=problem["gu"]
    )

    cpu_time = time.perf_counter() - start

    return {
        "solver": "Line-Search SQP",
        "objective": f_star,
        "iterations": len(history["f"]),
        "cpu_time": cpu_time,
        "success": True
    }


def run_trust_region_sqp(problem):
    start = time.perf_counter()

    x_star, f_star, history = trust_region_sqp(
        f_fun=problem["f"],
        grad_f_fun=problem["grad_f"],
        hess_f_fun=problem["hess_f"],
        x0=problem["x0"],
        xl=problem["xl"],
        xu=problem["xu"],
        g_fun=problem["g"],
        jac_g_fun=problem["jac_g"],
        gl=problem["gl"],
        gu=problem["gu"]
    )

    cpu_time = time.perf_counter() - start

    return {
        "solver": "Trust-Region SQP",
        "objective": f_star,
        "iterations": len(history["f"]),
        "cpu_time": cpu_time,
        "success": True
    }


if __name__ == "__main__":

    problems = {
        "Constrained Himmelblau": {
            "f": himmelblau,
            "grad_f": grad_himmelblau,
            "hess_f": hess_himmelblau,
            "x0": np.array([0.0, 0.0]),
            "xl": np.array([-5.0, -5.0]),
            "xu": np.array([5.0, 5.0]),
            "g": c_himmelblau,
            "jac_g": jac_c_himmelblau,
            "gl": np.array([0.0, 0.0]),
            "gu": np.array([BIG, BIG])
        }
    }

    solvers = [
        solve_with_library,
        run_line_search_sqp,
        run_trust_region_sqp
    ]

    all_results = []

    for problem_name, problem in problems.items():

        print(f"\nRunning problem: {problem_name}")
        print("=" * 60)

        for solver in solvers:

            result = solver(problem)

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

    plt.figure(figsize=(8, 5))

    plt.bar(
        df["solver"],
        df["cpu_time"]
    )

    plt.ylabel("CPU Time [s]")
    plt.title("NLP Solver CPU Time Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()

    plt.savefig(
        f"{FIGURE_DIR}/nlp_solver_cpu_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()