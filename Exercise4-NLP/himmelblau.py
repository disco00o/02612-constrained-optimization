import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from primal_dual_interior_point import primal_dual_interior_point
from scipy import sparse

from scipy.optimize import minimize, Bounds

from SQP import line_search_sqp, tr_sqp

# ---------------------------------------------------------
# Objective
# ---------------------------------------------------------
def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def grad_himmelblau(x):
    x1, x2 = x
    return np.array([
        4*x1*(x1**2 + x2 - 11) + 2*(x1 + x2**2 - 7),
        2*(x1**2 + x2 - 11) + 4*x2*(x1 + x2**2 - 7)
    ])

def hess_himmelblau(x):
    x1, x2 = x

    return np.array([
        [12 * x1**2 + 4 * x2 - 42,
         4 * x1 + 4 * x2],

        [4 * x1 + 4 * x2,
         4 * x1 + 12 * x2**2 - 26]
    ])

def hess_c_himmelblau(x):
    # Hessian of c1(x) = (x1+2)^2 - x2
    Hc1 = np.array([
        [2.0, 0.0],
        [0.0, 0.0]
    ])

    # Hessian of c2(x) = -4*x1 + 10*x2  
    Hc2 = np.zeros((2, 2))

    return np.array([Hc1, Hc2])


# ---------------------------------------------------------
# Constraints in the form gl =< g(x) =< gu
# ---------------------------------------------------------
def g_fun(x):
    x1, x2 = x
    return np.array([
        (x1 + 2)**2 - x2,     # c1(x) >= 0
        -4*x1 + 10*x2         # c2(x) >= 0
    ])

def c_himmelblau(x):
    x1, x2 = x
    c1 = (x1 + 2)**2 - x2          # => 0
    c2 = -4*x1 + 10*x2             # => 0
    return np.array([c1, c2])


def jac_c_himmelblau(x):
    x1, x2 = x
    # dc1/dx = [2(x1+2), -1]
    # dc2/dx = [-4, 10]
    return np.array([
        [2*(x1 + 2), -1.0],
        [-4.0,       10.0]
    ])

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

    # --- Himmelblau contours ---
    x1 = np.linspace(-5,5,400)
    x2 = np.linspace(-5,5,400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X1**2 + X2 - 11)**2 + (X1 + X2**2 - 7)**2
    plt.contour(X1, X2, Z, levels=50, cmap="viridis")

    # --- Compute infeasible region ---
    c1 = (X1 + 2)**2 - X2
    c2 = -4*X1 + 10*X2

    infeasible = (c1 < 0) | (c2 < 0)

    # --- Shade infeasible region in red ---
    plt.imshow(
        infeasible.astype(int),
        extent=[-5, 5, -5, 5],
        origin="lower",
        alpha=0.25,
        cmap="Reds"
    )

    # --- Plot paths ---
    plt.plot(xs_exact[:,0], xs_exact[:,1], "o-", label="Exact Hessian")
    plt.plot(xs_bfgs[:,0],  xs_bfgs[:,1],  "s-", label="BFGS")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Optimization Paths with Infeasible Region (Red)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import time


if __name__ == "__main__":
    xl = np.array([-5.0, -5.0])
    xu = np.array([5.0, 5.0])
    x0 = np.array([0, 0])

    # --- 1. Trust Region SQP (Exact Hessian) ---
    start_tr_exact = time.perf_counter()
    x_tr_exact, f_tr_exact, hist_tr_exact = tr_sqp(
        obj_fun=himmelblau, 
        grad_obj_fun=grad_himmelblau, 
        hess_obj_fun=hess_himmelblau, 
        hess_c_obj_fun=hess_c_himmelblau, 
        c_obj_fun=c_himmelblau, 
        jac_c_obj_fun=jac_c_himmelblau,
        x0=x0, xl=xl, xu=xu, hessian_mode="exact"
    )
    time_tr_exact = time.perf_counter() - start_tr_exact

    # --- 2. Trust Region SQP (BFGS Hessian) ---
    start_tr_bfgs = time.perf_counter()
    x_tr_bfgs, f_tr_bfgs, hist_tr_bfgs = tr_sqp(
        obj_fun=himmelblau, 
        grad_obj_fun=grad_himmelblau, 
        hess_obj_fun=hess_himmelblau, 
        hess_c_obj_fun=hess_c_himmelblau, 
        c_obj_fun=c_himmelblau, 
        jac_c_obj_fun=jac_c_himmelblau,
        x0=x0, xl=xl, xu=xu, hessian_mode="bfgs", max_iter=20
    )
    time_tr_bfgs = time.perf_counter() - start_tr_bfgs

    # Plot Trust Region
    plot_convergence_compare(hist_tr_exact, hist_tr_bfgs)
    plot_paths(hist_tr_exact, hist_tr_bfgs)

    # --- 3. Line Search SQP (Exact Hessian) ---
    start_ls_exact = time.perf_counter()
    x_ls_exact, f_ls_exact, hist_ls_exact = line_search_sqp(
        obj_fun=himmelblau, 
        grad_obj_fun=grad_himmelblau, 
        hess_obj_fun=hess_himmelblau, 
        hess_c_obj_fun=hess_c_himmelblau, 
        c_obj_fun=c_himmelblau, 
        jac_c_obj_fun=jac_c_himmelblau,
        x0=x0, xl=xl, xu=xu, hessian_mode="exact"
    )
    time_ls_exact = time.perf_counter() - start_ls_exact

    # --- 4. Line Search SQP (BFGS Hessian) ---
    start_ls_bfgs = time.perf_counter()
    x_ls_bfgs, f_ls_bfgs, hist_ls_bfgs = line_search_sqp(
        obj_fun=himmelblau, 
        grad_obj_fun=grad_himmelblau, 
        hess_obj_fun=hess_himmelblau, 
        hess_c_obj_fun=hess_c_himmelblau, 
        c_obj_fun=c_himmelblau, 
        jac_c_obj_fun=jac_c_himmelblau,
        x0=x0, xl=xl, xu=xu, hessian_mode="bfgs"
    )
    time_ls_bfgs = time.perf_counter() - start_ls_bfgs

    # Plot Line Search
    plot_convergence_compare(hist_ls_exact, hist_ls_bfgs)
    plot_paths(hist_ls_exact, hist_ls_bfgs)

    # --- Print Results and Timing Summary ---
    print("\n" + "="*60)
    print("                SOLVER BENCHMARK RESULTS")
    print("="*60)
    
    print(f"\n[Trust Region SQP - Exact Hessian]")
    print(f"  Optimal x: {x_tr_exact}")
    print(f"  F-val    : {f_tr_exact}")
    print(f"  Runtime  : {time_tr_exact:.6f} seconds")
    
    print(f"\n[Trust Region SQP - BFGS]")
    print(f"  Optimal x: {x_tr_bfgs}")
    print(f"  F-val    : {f_tr_bfgs}")
    print(f"  Runtime  : {time_tr_bfgs:.6f} seconds")
    
    print(f"\n[Line Search SQP - Exact Hessian]")
    print(f"  Optimal x: {x_ls_exact}")
    print(f"  F-val    : {f_ls_exact}")
    print(f"  Runtime  : {time_ls_exact:.6f} seconds")
    
    print(f"\n[Line Search SQP - BFGS]")
    print(f"  Optimal x: {x_ls_bfgs}")
    print(f"  F-val    : {f_ls_bfgs}")
    print(f"  Runtime  : {time_ls_bfgs:.6f} seconds")
    print("="*60)
