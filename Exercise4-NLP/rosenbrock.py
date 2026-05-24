import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from primal_dual_interior_point import primal_dual_interior_point
from scipy import sparse

from scipy.optimize import minimize, Bounds

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
    xu,
    gk,   # g(xk)
    Jgk,  # J_g(xk)
    gl,
    gu
):
    n = len(xk)
    gk = np.asarray(gk, float)
    gl = np.asarray(gl, float)
    gu = np.asarray(gu, float)

    
    # bounds on p from box constraints
    p_lower = xl - xk
    p_upper = xu - xk
    
    A = sparse.csr_matrix(Jgk.T)  # (n, m)

    # b_lower = g_l - g(xk), b_upper = g_u - g(xk)
    # use large finite values for "infinite" bounds
    big = 1e20
    b_lower = np.where(np.isfinite(gl), gl - gk, -big)
    b_upper = np.where(np.isfinite(gu), gu - gk,  big)

    pk, z_ipm, s_ipm, obj_val, *_ = primal_dual_interior_point(
        H=Hk,
        g=grad_fk,
        A=A,
        b_lower=b_lower,
        b_upper=b_upper,
        x_lower=p_lower,
        x_upper=p_upper,
        x0=np.zeros_like(xk),
    )

    # --- decode multipliers 
    m = len(gk)
    n = len(xk)

    idx_g  = slice(0, m)
    idx_xl = slice(m, m + n)
    idx_xu = slice(m + n, m + 2*n)

    lambda_k = z_ipm[idx_g]
    mu_l_k   = z_ipm[idx_xl]
    mu_u_k   = z_ipm[idx_xu]

    
    print("IPM step:  ", pk)
    #print("---------------------\n")

    return pk, lambda_k, mu_l_k, mu_u_k




def constraint_violation(g):
    return np.sum(np.maximum(0.0, -g))

def merit_function(f, g, rho):
    return f + rho * constraint_violation(g)

def line_search_sqp(
    x0,
    xl,
    xu,
    gl=[0.0, 0.0],
    gu=[np.inf, np.inf],
    hessian_mode="exact",
    max_iter=100,
    tol=1e-8
):
    """
    Line-search SQP for:
        min f(x)
        s.t. gl <= g(x) <= gu
             xl <= x <= xu
    """

    history = {"f": [], "x": []}

    xk = np.asarray(x0, dtype=float)
    xl = np.asarray(xl, dtype=float)
    xu = np.asarray(xu, dtype=float)

    gl = np.asarray(gl, dtype=float)
    gu = np.asarray(gu, dtype=float)

    n = len(xk)

    # masks for finite lower/upper bounds on g
    mask_l = np.isfinite(gl)
    mask_u = np.isfinite(gu)

    Bk = np.eye(n)

    lambda_k = np.zeros(0)
    mu_l_k   = np.zeros_like(xk)
    mu_u_k   = np.zeros_like(xk)

    for k in range(max_iter):

        fk = himmelblau(xk)
        grad_fk = grad_himmelblau(xk)

        gk_raw = c_himmelblau(xk)      # g(xk), shape (m,)
        Jg_raw = jac_c_himmelblau(xk)  # (m, n)
        Hc_raw = hess_c_himmelblau(xk) # list/array of H_i

        # build inequality system c(x) >= 0 from gl <= g(x) <= gu
        c_list = []
        J_list = []

        idx_l = np.where(mask_l)[0]
        if idx_l.size > 0:
            c_list.append(gk_raw[idx_l] - gl[idx_l])
            J_list.append(Jg_raw[idx_l, :])

        idx_u = np.where(mask_u)[0]
        if idx_u.size > 0:
            c_list.append(gu[idx_u] - gk_raw[idx_u])
            J_list.append(-Jg_raw[idx_u, :])

        if len(c_list) > 0:
            ck = np.concatenate(c_list)
            Jk = np.vstack(J_list)
        else:
            ck = np.zeros(0)
            Jk = np.zeros((0, n))

        history["x"].append(xk.copy())
        history["f"].append(fk)

        # stationarity check (using Lagrangian gradient)
        if lambda_k.size == 0:
            grad_Lk = grad_fk.copy()
        else:
            grad_Lk = grad_fk + Jk.T @ lambda_k

        if mu_l_k is not None and mu_l_k.size > 0:
            grad_Lk = grad_Lk + mu_l_k
        if mu_u_k is not None and mu_u_k.size > 0:
            grad_Lk = grad_Lk - mu_u_k

        if np.linalg.norm(grad_Lk, np.inf) < tol:
            break

        # -------------------------------
        # Hessian of the Lagrangian
        # -------------------------------
        if hessian_mode == "exact":
            Hk = hess_himmelblau(xk)

            if lambda_k.size > 0:
                offset = 0
                # lower bounds: +lambda * H_i
                for j, i in enumerate(idx_l):
                    Hk += lambda_k[offset + j] * Hc_raw[i]
                offset += len(idx_l)
                # upper bounds: +lambda * (-H_i)
                for j, i in enumerate(idx_u):
                    Hk += lambda_k[offset + j] * (-Hc_raw[i])
        else:
            Hk = Bk

        # ensure PD
        eig_min = np.min(np.linalg.eigvalsh(Hk))
        if eig_min <= 1e-8:
            Hk += (abs(eig_min) + 1e-4) * np.eye(n)

        # -------------------------------
        # QP subproblem
        # -------------------------------
       
        gk = c_himmelblau(xk)        # shape (m,)
        Jgk = jac_c_himmelblau(xk)   # shape (m,n)

        pk, lambda_k, mu_l_k, mu_u_k = solve_qp_subproblem(
            xk=xk,
            Hk=Hk,
            grad_fk=grad_fk,
            xl=xl,          # or xk+p_lower_tr for TR
            xu=xu,          # or xk+p_upper_tr for TR
            gk=gk,
            Jgk=Jgk,
            gl=gl,          
            gu=gu           
        )

        # -------------------------------
        # Line search (l1 merit)
        # -------------------------------
        rho = 10.0
        alpha = 1.0
        rho_backtrack = 0.5
        c1 = 1e-4

        phi_k = merit_function(fk, ck, rho)

        while alpha > 1e-12:
            x_trial = xk + alpha * pk
            x_trial = np.minimum(np.maximum(x_trial, xl), xu)

            f_trial = himmelblau(x_trial)
            g_trial_raw = c_himmelblau(x_trial)

            c_trial_list = []
            if idx_l.size > 0:
                c_trial_list.append(g_trial_raw[idx_l] - gl[idx_l])
            if idx_u.size > 0:
                c_trial_list.append(gu[idx_u] - g_trial_raw[idx_u])

            if len(c_trial_list) > 0:
                g_trial = np.concatenate(c_trial_list)
            else:
                g_trial = np.zeros(0)

            phi_trial = merit_function(f_trial, g_trial, rho)

            # simple directional derivative model 
            Dphi = grad_fk @ pk - rho * np.linalg.norm(ck, 1)

            if phi_trial <= phi_k + c1 * alpha * Dphi:
                break

            alpha *= rho_backtrack

        x_next = x_trial

        # -------------------------------
        # BFGS update (Lagrangian)
        # -------------------------------
        if hessian_mode == "bfgs":

            grad_f_next = grad_himmelblau(x_next)
            g_next_raw = c_himmelblau(x_next)
            J_next_raw = jac_c_himmelblau(x_next)

            J_next_list = []
            if idx_l.size > 0:
                J_next_list.append(J_next_raw[idx_l, :])
            if idx_u.size > 0:
                J_next_list.append(-J_next_raw[idx_u, :])

            if len(J_next_list) > 0:
                J_next = np.vstack(J_next_list)
            else:
                J_next = np.zeros((0, n))

            # grad L at xk
            if lambda_k.size == 0:
                grad_Lk = grad_fk.copy()
            else:
                grad_Lk = grad_fk + Jk.T @ lambda_k
            if mu_l_k is not None and mu_l_k.size > 0:
                grad_Lk = grad_Lk + mu_l_k
            if mu_u_k is not None and mu_u_k.size > 0:
                grad_Lk = grad_Lk - mu_u_k

            # grad L at x_next
            if lambda_k.size == 0:
                grad_L_next = grad_f_next.copy()
            else:
                grad_L_next = grad_f_next + J_next.T @ lambda_k
            if mu_l_k is not None and mu_l_k.size > 0:
                grad_L_next = grad_L_next + mu_l_k
            if mu_u_k is not None and mu_u_k.size > 0:
                grad_L_next = grad_L_next - mu_u_k

            s = x_next - xk
            y = grad_L_next - grad_Lk

            Bk = damped_bfgs_update(Bk, s, y)

        # stopping on step size
        if np.linalg.norm(x_next - xk) < tol:
            xk = x_next
            break

        xk = x_next

    return xk, himmelblau(xk), history


def tr_sqp(
    x0,
    xl,
    xu,
    gl=[0.0, 0.0],
    gu=[np.inf,np.inf],
    hessian_mode="exact",
    max_iter=100,
    tol_opt=1e-6,
    tol_feas=1e-6,
    Delta0=1.0,
    Delta_max=10.0,
    eta1=0.25,
    eta2=0.75,
    gamma_dec=0.25,
    gamma_inc=2,
    rho_merit=10.0
):
    """
    Trust-region SQP for:
        min f(x)
        s.t. gl <= g(x) <= gu
             xl <= x <= xu
    """

    history = {"x": [], "f": [], "Delta": [], "rho": []}

    xk = np.asarray(x0, float)
    xl = np.asarray(xl, float)
    xu = np.asarray(xu, float)

    gl = np.asarray(gl, float)
    gu = np.asarray(gu, float)

    n = len(xk)

    # masks for finite lower/upper bounds on g
    mask_l = np.isfinite(gl)
    mask_u = np.isfinite(gu)

    Bk = np.eye(n)
    Delta_k = Delta0

    lambda_k = None   # multipliers for nonlinear constraints (stacked)
    mu_l_k = None     # multipliers for lower box bounds on x
    mu_u_k = None     # multipliers for upper box bounds on x

    for k in range(max_iter):

        fk = himmelblau(xk)
        grad_fk = grad_himmelblau(xk)

        # raw constraints g(x)
        gk_raw = c_himmelblau(xk)      # shape (m,)
        Jg_raw = jac_c_himmelblau(xk)  # shape (m, n)
        Hc_raw = hess_c_himmelblau(xk) # list/array of Hessians H_i

        # build inequality system c(x) >= 0 from gl <= g(x) <= gu
        c_list = []
        J_list = []

        # lower bounds: g_i(x) - gl_i >= 0
        idx_l = np.where(mask_l)[0]
        if idx_l.size > 0:
            c_list.append(gk_raw[idx_l] - gl[idx_l])
            J_list.append(Jg_raw[idx_l, :])

        # upper bounds: gu_i - g_i(x) >= 0
        idx_u = np.where(mask_u)[0]
        if idx_u.size > 0:
            c_list.append(gu[idx_u] - gk_raw[idx_u])
            J_list.append(-Jg_raw[idx_u, :])

        if len(c_list) > 0:
            ck = np.concatenate(c_list)
            Jk = np.vstack(J_list)
        else:
            ck = np.zeros(0)
            Jk = np.zeros((0, n))

        history["x"].append(xk.copy())
        history["f"].append(fk)
        history["Delta"].append(Delta_k)

        # Lagrangian gradient for KKT stopping
        if lambda_k is None or lambda_k.size == 0:
            grad_Lk = grad_fk.copy()
        else:
            grad_Lk = grad_fk + Jk.T @ lambda_k

        # add box multipliers safely
        if mu_l_k is not None and mu_l_k.size > 0:
            grad_Lk = grad_Lk + mu_l_k
        if mu_u_k is not None and mu_u_k.size > 0:
            grad_Lk = grad_Lk - mu_u_k

        # feasibility: violation of c(x) >= 0 -> min(c_i, 0)
        viol = np.minimum(ck, 0.0) if ck.size > 0 else np.array([0.0])
        feas_norm = np.linalg.norm(viol, np.inf)
        opt_norm = np.linalg.norm(grad_Lk, np.inf)

        if feas_norm < tol_feas and opt_norm < tol_opt:
            break

        # Hessian of Lagrangian
        if hessian_mode == "exact":
            Hk = hess_himmelblau(xk)

            if lambda_k is not None and lambda_k.size > 0:
                # ordering in ck: [lower(idx_l); upper(idx_u)]
                offset = 0
                # lower bounds: +lambda * H_i
                for j, i in enumerate(idx_l):
                    Hk += lambda_k[offset + j] * Hc_raw[i]
                offset += len(idx_l)
                # upper bounds: +lambda * (-H_i)
                for j, i in enumerate(idx_u):
                    Hk += lambda_k[offset + j] * (-Hc_raw[i])
        else:
            Hk = Bk

        # regularize to be positive definite
        eig_min = np.min(np.linalg.eigvalsh(Hk))
        if eig_min <= 1e-8:
            Hk += (abs(eig_min) + 1e-4) * np.eye(n)

        # trust-region bounds merged with box bounds
        p_lower_tr = np.maximum(xl - xk, -Delta_k * np.ones(n))
        p_upper_tr = np.minimum(xu - xk,  Delta_k * np.ones(n))

        # QP subproblem (TR handled via tightened bounds)
        pk_tr, lambda_k, mu_l_k, mu_u_k = solve_qp_subproblem(
            xk=xk,
            Hk=Hk,
            grad_fk=grad_fk,
            xl=xk + p_lower_tr,   # tightened bounds
            xu=xk + p_upper_tr,
            gk=ck,
            Jgk=Jk,
            gl=gl,
            gu=gu
        )

        # Merit at current point
        phi_k = merit_function(fk, ck, rho_merit)

        # Predicted reduction
        f_model = fk + grad_fk @ pk_tr + 0.5 * pk_tr @ (Hk @ pk_tr)
        c_model = ck + Jk @ pk_tr if ck.size > 0 else ck
        phi_model = merit_function(f_model, c_model, rho_merit)
        pred_red = phi_k - phi_model

        # Actual reduction
        x_trial = np.clip(xk + pk_tr, xl, xu)

        f_trial = himmelblau(x_trial)
        g_trial_raw = c_himmelblau(x_trial)

        c_trial_list = []
        # lower
        if idx_l.size > 0:
            c_trial_list.append(g_trial_raw[idx_l] - gl[idx_l])
        # upper
        if idx_u.size > 0:
            c_trial_list.append(gu[idx_u] - g_trial_raw[idx_u])

        if len(c_trial_list) > 0:
            c_trial = np.concatenate(c_trial_list)
        else:
            c_trial = np.zeros(0)

        phi_trial = merit_function(f_trial, c_trial, rho_merit)
        ared = phi_k - phi_trial

        rho_k = 0.0 if pred_red <= 0 else ared / pred_red
        history["rho"].append(rho_k)

        # trust-region update
        if rho_k < 0:
            Delta_k = max(1e-12, gamma_dec * Delta_k)
            x_next = xk
        elif rho_k < eta1:
            Delta_k = max(1e-12, gamma_dec * Delta_k)
            x_next = x_trial
        else:
            x_next = x_trial

            if rho_k > eta2 and abs(np.linalg.norm(pk_tr) - Delta_k) < 1e-6:
                Delta_k = min(gamma_inc * Delta_k, Delta_max)

            # BFGS update
            if hessian_mode == "bfgs":
                grad_f_next = grad_himmelblau(x_next)
                g_next_raw = c_himmelblau(x_next)
                J_next_raw = jac_c_himmelblau(x_next)

                J_next_list = []
                if idx_l.size > 0:
                    J_next_list.append(J_next_raw[idx_l, :])
                if idx_u.size > 0:
                    J_next_list.append(-J_next_raw[idx_u, :])

                if len(J_next_list) > 0:
                    J_next = np.vstack(J_next_list)
                else:
                    J_next = np.zeros((0, n))

                # grad L at xk
                if lambda_k is None or lambda_k.size == 0:
                    grad_Lk = grad_fk.copy()
                else:
                    grad_Lk = grad_fk + Jk.T @ lambda_k
                if mu_l_k is not None and mu_l_k.size > 0:
                    grad_Lk = grad_Lk + mu_l_k
                if mu_u_k is not None and mu_u_k.size > 0:
                    grad_Lk = grad_Lk - mu_u_k

                # grad L at x_next
                if lambda_k is None or lambda_k.size == 0:
                    grad_L_next = grad_f_next.copy()
                else:
                    grad_L_next = grad_f_next + J_next.T @ lambda_k
                if mu_l_k is not None and mu_l_k.size > 0:
                    grad_L_next = grad_L_next + mu_l_k
                if mu_u_k is not None and mu_u_k.size > 0:
                    grad_L_next = grad_L_next - mu_u_k

                s = x_next - xk
                y = grad_L_next - grad_Lk

                Bk = damped_bfgs_update(Bk, s, y)

        xk = x_next

    return xk, himmelblau(xk), history


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

    # Hessian of c2(x) = -4*x1 + 10*x2  (affine → zero Hessian)
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

import time
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Plot helpers (unchanged, but now used for Rosenbrock)
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

# ---------------------------------------------------------
# Main: run TR-SQP and LS-SQP on Rosenbrock
# ---------------------------------------------------------
if __name__ == "__main__":
    xl = np.array([-2.0, -1.0])
    xu = np.array([ 2.0,  3.0])
    x0 = np.array([-1.2, 1.0])

    gl = np.array([0.0, 0.0])
    gu = np.array([np.inf, np.inf])

    # hook Rosenbrock into existing SQP code
    himmelblau        = rosenbrock
    grad_himmelblau   = grad_rosenbrock
    hess_himmelblau   = hess_rosenbrock
    c_himmelblau      = c_rosenbrock
    jac_c_himmelblau  = jac_c_rosenbrock
    hess_c_himmelblau = hess_c_rosenbrock

    # --- 1. Trust Region SQP (Exact) ---
    start_tr_exact = time.perf_counter()
    x_tr_exact, f_tr_exact, hist_tr_exact = tr_sqp(
        x0=x0, xl=xl, xu=xu, gl=gl, gu=gu, hessian_mode="exact"
    )
    time_tr_exact = time.perf_counter() - start_tr_exact

    # --- 2. Trust Region SQP (BFGS) ---
    start_tr_bfgs = time.perf_counter()
    x_tr_bfgs, f_tr_bfgs, hist_tr_bfgs = tr_sqp(
        x0=x0, xl=xl, xu=xu, gl=gl, gu=gu, hessian_mode="bfgs"
    )
    time_tr_bfgs = time.perf_counter() - start_tr_bfgs

    # --- 3. Line Search SQP (Exact) ---
    start_ls_exact = time.perf_counter()
    x_ls_exact, f_ls_exact, hist_ls_exact = line_search_sqp(
        x0=x0, xl=xl, xu=xu, gl=gl, gu=gu, hessian_mode="exact"
    )
    time_ls_exact = time.perf_counter() - start_ls_exact

    # --- 4. Line Search SQP (BFGS) ---
    start_ls_bfgs = time.perf_counter()
    x_ls_bfgs, f_ls_bfgs, hist_ls_bfgs = line_search_sqp(
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



