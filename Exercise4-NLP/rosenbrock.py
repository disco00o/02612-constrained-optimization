import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from SQP import line_search_sqp, tr_sqp
from scipy.optimize import minimize, Bounds
import time


# ---------------------------------------------------------
# Plot helpers 
# ---------------------------------------------------------
def plot_convergence_compare(hist_exact, hist_bfgs):
    plt.figure(figsize=(7,5))

    plt.semilogy(hist_exact["f"], label="Exact Hessian", marker="o")
    plt.semilogy(hist_bfgs["f"], label="BFGS", marker="s")

    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.title("SQP Convergence Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_paths(hist_exact, hist_bfgs):
    xs_exact = np.array(hist_exact["x"])
    xs_bfgs  = np.array(hist_bfgs["x"])

    plt.figure(figsize=(7,6))

    # --- Rosenbrock contours ---
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (1 - X1)**2 + 100*(X2 - X1**2)**2
    plt.contour(X1, X2, Z, levels=50, cmap="viridis")

    # --- Constraints (same as before) ---
    c1 = (X1 + 2)**2 - X2
    c2 = -4*X1 + 10*X2
    infeasible = (c1 < 0) | (c2 < 0)

    plt.imshow(
        infeasible.astype(int),
        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
        origin="lower",
        alpha=0.25,
        cmap="Reds"
    )

    # --- Paths ---
    plt.plot(xs_exact[:,0], xs_exact[:,1], "o-", label="Exact Hessian")
    plt.plot(xs_bfgs[:,0],  xs_bfgs[:,1],  "s-", label="BFGS")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Optimization Paths with Infeasible Region (Red)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Rosenbrock + constraints
# ---------------------------------------------------------
def rosenbrock(x, a=1.0, b=100.0):
    x1, x2 = x
    return (a - x1)**2 + b * (x2 - x1**2)**2


def grad_rosenbrock(x, a=1.0, b=100.0):
    x1, x2 = x
    return np.array([
        -2*(a - x1) - 4*b*x1*(x2 - x1**2),
        2*b*(x2 - x1**2)
    ])


def hess_rosenbrock(x, a=1.0, b=100.0):
    x1, x2 = x
    return np.array([
        [2 - 4*b*(x2 - x1**2) + 8*b*x1**2,  -4*b*x1],
        [-4*b*x1,                           2*b    ]
    ])


def c_rosenbrock(x):
    x1, x2 = x
    c1 = (x1 + 2)**2 - x2
    c2 = -4*x1 + 10*x2
    return np.array([c1, c2])


def jac_c_rosenbrock(x):
    x1, x2 = x
    return np.array([
        [2*(x1 + 2), -1.0],
        [-4.0,        10.0]
    ])


def hess_c_rosenbrock(x):
    Hc1 = np.array([[2.0, 0.0],
                    [0.0, 0.0]])
    Hc2 = np.zeros((2, 2))
    return np.array([Hc1, Hc2])

import casadi as ca
import numpy as np

def solve_with_ipopt(x0, gl, gu):
    x = ca.MX.sym('x', 2)

    # Objective
    f = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

    # Constraints
    g1 = (x[0] + 2)**2 - x[1]
    g2 = -4*x[0] + 10*x[1]
    g = ca.vertcat(g1, g2)

    nlp = {'x': x, 'f': f, 'g': g}

    solver = ca.nlpsol('solver', 'ipopt', nlp, {
        'ipopt.print_level': 0,
        'print_time': False
    })

    sol = solver(
        x0=x0,
        lbx=[-2, -1],
        ubx=[ 2,  3],
        lbg=gl,
        ubg=gu
    )

    return np.array(sol['x']).flatten(), float(sol['f'])

from scipy.optimize import minimize, NonlinearConstraint, Bounds

def solve_with_trust_constr(x0):
    def f(x):
        return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

    def g(x):
        return np.array([
            (x[0]+2)**2 - x[1],
            -4*x[0] + 10*x[1]
        ])

    def Jg(x):
        return np.array([
            [2*(x[0]+2), -1],
            [-4, 10]
        ])

    cons = NonlinearConstraint(g, [0,0], [np.inf, np.inf], jac=Jg)
    bounds = Bounds([-2,-1], [2,3])

    res = minimize(f, x0, method='trust-constr', constraints=[cons], bounds=bounds)
    return res.x, res.fun

def plot_cpu_times(time_tr_exact, time_tr_bfgs, time_ls_exact, time_ls_bfgs):
    labels = [
        "TR-SQP (Exact)",
        "TR-SQP (BFGS)",
        "LS-SQP (Exact)",
        "LS-SQP (BFGS)"
    ]
    times = [
        time_tr_exact,
        time_tr_bfgs,
        time_ls_exact,
        time_ls_bfgs
    ]

    plt.figure(figsize=(7,5))
    bars = plt.bar(labels, times, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height,
            f"{height:.4f}s",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.ylabel("CPU Time [s]")
    plt.title("Solver Runtime Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def plot_cpu_times_extended(times_dict):
    labels = list(times_dict.keys())
    times  = list(times_dict.values())

    plt.figure(figsize=(9,5))
    bars = plt.bar(labels, times,
                   color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"])

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h,
                 f"{h:.4f}s", ha="center", va="bottom", fontsize=9)

    plt.ylabel("CPU Time [s]")
    plt.title("Solver Runtime Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

def plot_final_solutions_extended(solutions):
    plt.figure(figsize=(7,6))

    # Rosenbrock contours
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (1 - X1)**2 + 100*(X2 - X1**2)**2
    plt.contour(X1, X2, Z, levels=50, cmap="viridis")

    # Constraints
    c1 = (X1 + 2)**2 - X2
    c2 = -4*X1 + 10*X2
    infeasible = (c1 < 0) | (c2 < 0)

    plt.imshow(
        infeasible.astype(int),
        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
        origin="lower",
        alpha=0.25,
        cmap="Reds"
    )

    # Plot all solver solutions
    markers = ["o", "s", "D", "^", "P", "X"]
    for (label, x), m in zip(solutions.items(), markers):
        plt.plot(x[0], x[1], m, markersize=10, label=label)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Final Solutions of All Solvers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_final_solutions(solutions):
    """
    solutions = {
        "TR Exact": x_tr_exact,
        "TR BFGS": x_tr_bfgs,
        "LS Exact": x_ls_exact,
        "LS BFGS": x_ls_bfgs
    }
    """
    plt.figure(figsize=(7,6))

    # Rosenbrock contours
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (1 - X1)**2 + 100*(X2 - X1**2)**2
    plt.contour(X1, X2, Z, levels=50, cmap="viridis")

    # Constraints
    c1 = (X1 + 2)**2 - X2
    c2 = -4*X1 + 10*X2
    infeasible = (c1 < 0) | (c2 < 0)

    plt.imshow(
        infeasible.astype(int),
        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
        origin="lower",
        alpha=0.25,
        cmap="Reds"
    )

    # Plot all solver solutions
    markers = ["o", "s", "D", "^"]
    for (label, x), m in zip(solutions.items(), markers):
        plt.plot(x[0], x[1], m, markersize=10, label=label)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Final Solutions of All Solvers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    xl = np.array([-2.0, -1.0])
    xu = np.array([ 2.0,  3.0])
    x0 = np.array([-1.2, 1.0])

    gl = np.array([0.0, 0.0])
    gu = np.array([np.inf, np.inf])

    
    # --- 1. Trust Region SQP (Exact) ---
    start_tr_exact = time.perf_counter()
    x_tr_exact, f_tr_exact, hist_tr_exact = tr_sqp(
        obj_fun=rosenbrock, 
        grad_obj_fun=grad_rosenbrock, 
        hess_obj_fun=hess_rosenbrock, 
        hess_c_obj_fun=hess_c_rosenbrock, 
        c_obj_fun=c_rosenbrock, 
        jac_c_obj_fun=jac_c_rosenbrock,
        x0=x0, xl=xl, xu=xu, gl=gl, gu=gu, hessian_mode="exact"
    )
    time_tr_exact = time.perf_counter() - start_tr_exact

    # --- 2. Trust Region SQP (BFGS) ---
    start_tr_bfgs = time.perf_counter()
    x_tr_bfgs, f_tr_bfgs, hist_tr_bfgs = tr_sqp(
        obj_fun=rosenbrock, 
        grad_obj_fun=grad_rosenbrock, 
        hess_obj_fun=hess_rosenbrock, 
        hess_c_obj_fun=hess_c_rosenbrock, 
        c_obj_fun=c_rosenbrock, 
        jac_c_obj_fun=jac_c_rosenbrock,
        x0=x0, xl=xl, xu=xu, gl=gl, gu=gu, hessian_mode="bfgs"
    )
    time_tr_bfgs = time.perf_counter() - start_tr_bfgs

    # --- 3. Line Search SQP (Exact) ---
    start_ls_exact = time.perf_counter()
    x_ls_exact, f_ls_exact, hist_ls_exact = line_search_sqp(
        obj_fun=rosenbrock, 
        grad_obj_fun=grad_rosenbrock, 
        hess_obj_fun=hess_rosenbrock, 
        hess_c_obj_fun=hess_c_rosenbrock, 
        c_obj_fun=c_rosenbrock, 
        jac_c_obj_fun=jac_c_rosenbrock,
        x0=x0, xl=xl, xu=xu, gl=gl, gu=gu, hessian_mode="exact"
    )
    time_ls_exact = time.perf_counter() - start_ls_exact

    # --- 4. Line Search SQP (BFGS) ---
    start_ls_bfgs = time.perf_counter()
    x_ls_bfgs, f_ls_bfgs, hist_ls_bfgs = line_search_sqp(
        obj_fun=rosenbrock, 
        grad_obj_fun=grad_rosenbrock, 
        hess_obj_fun=hess_rosenbrock, 
        hess_c_obj_fun=hess_c_rosenbrock, 
        c_obj_fun=c_rosenbrock, 
        jac_c_obj_fun=jac_c_rosenbrock,
        x0=x0, xl=xl, xu=xu, gl=gl, gu=gu, hessian_mode="bfgs"
    )
    time_ls_bfgs = time.perf_counter() - start_ls_bfgs

    # --- 5. IPOPT ---
    start_ipopt = time.perf_counter()
    x_ipopt, f_ipopt = solve_with_ipopt(x0, gl, gu)
    time_ipopt = time.perf_counter() - start_ipopt

    # --- 6. trust-constr ---
    start_tc = time.perf_counter()
    x_tc, f_tc = solve_with_trust_constr(x0)
    time_tc = time.perf_counter() - start_tc


    # --- Plots ---
    print("\nTrust Region SQP (Rosenbrock)")
    plot_convergence_compare(hist_tr_exact, hist_tr_bfgs)
    plot_paths(hist_tr_exact, hist_tr_bfgs)

    print("\nLine Search SQP (Rosenbrock)")
    plot_convergence_compare(hist_ls_exact, hist_ls_bfgs)
    plot_paths(hist_ls_exact, hist_ls_bfgs)

    # --- Summary ---
    print("\n" + "="*60)
    print("                SOLVER BENCHMARK RESULTS (ROSENBROCK)")
    print("="*60)

    print(f"\n[Trust Region SQP - Exact Hessian]")
    print(f"  Optimal x: {x_tr_exact}")
    print(f"  F-val    : {f_tr_exact}")
    print(f"  Runtime  : {time_tr_exact:.6f} s")

    print(f"\n[Trust Region SQP - BFGS]")
    print(f"  Optimal x: {x_tr_bfgs}")
    print(f"  F-val    : {f_tr_bfgs}")
    print(f"  Runtime  : {time_tr_bfgs:.6f} s")

    print(f"\n[Line Search SQP - Exact Hessian]")
    print(f"  Optimal x: {x_ls_exact}")
    print(f"  F-val    : {f_ls_exact}")
    print(f"  Runtime  : {time_ls_exact:.6f} s")

    print(f"\n[Line Search SQP - BFGS]")
    print(f"  Optimal x: {x_ls_bfgs}")
    print(f"  F-val    : {f_ls_bfgs}")
    print(f"  Runtime  : {time_ls_bfgs:.6f} s")

    print(f"\n[IPOPT]")
    print(f"  Optimal x: {x_ipopt}")
    print(f"  F-val    : {f_ipopt}")
    print(f"  Runtime  : {time_ipopt:.6f} s")

    print(f"\n[trust-constr]")
    print(f"  Optimal x: {x_tc}")
    print(f"  F-val    : {f_tc}")
    print(f"  Runtime  : {time_tc:.6f} s")  

    print("="*60)

    solutions = {
    "TR Exact": x_tr_exact,
    "TR BFGS": x_tr_bfgs,
    "LS Exact": x_ls_exact,
    "LS BFGS": x_ls_bfgs
    }

    plot_final_solutions(solutions)
    plot_cpu_times(time_tr_exact, time_tr_bfgs, time_ls_exact, time_ls_bfgs)

    times_dict = {
    "TR-SQP Exact": time_tr_exact,
    "TR-SQP BFGS": time_tr_bfgs,
    "LS-SQP Exact": time_ls_exact,
    "LS-SQP BFGS": time_ls_bfgs,
    "IPOPT": time_ipopt,
    "trust-constr": time_tc
    }

    plot_cpu_times_extended(times_dict)
    solutions = {
        "TR Exact": x_tr_exact,
        "TR BFGS": x_tr_bfgs,
        "LS Exact": x_ls_exact,
        "LS BFGS": x_ls_bfgs,
        "IPOPT": x_ipopt,
        "trust-constr": x_tc
    }

    plot_final_solutions_extended(solutions)



