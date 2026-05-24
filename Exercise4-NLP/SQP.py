from primal_dual_interior_point import primal_dual_interior_point
from scipy import sparse
import numpy as np


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

    
    #print("IPM step:  ", pk)
    #print("---------------------\n")

    return pk, lambda_k, mu_l_k, mu_u_k




def constraint_violation(g):
    return np.sum(np.maximum(0.0, -g))

def merit_function(f, g, rho):
    return f + rho * constraint_violation(g)

def line_search_sqp(
    obj_fun,
    grad_obj_fun,
    c_obj_fun,
    jac_c_obj_fun,
    hess_c_obj_fun,
    x0,
    xl,
    xu,
    gl=[0.0, 0.0],
    gu=[np.inf, np.inf],
    hessian_mode="exact",
    hess_obj_fun=None,
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

        fk = obj_fun(xk)
        grad_fk = grad_obj_fun(xk)

        gk_raw = c_obj_fun(xk)      # g(xk), shape (m,)
        Jg_raw = jac_c_obj_fun(xk)  # (m, n)
        Hc_raw = hess_c_obj_fun(xk) # list/array of H_i

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
            Hk = hess_obj_fun(xk)

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
       
        gk = c_obj_fun(xk)        # shape (m,)
        Jgk = jac_c_obj_fun(xk)   # shape (m,n)

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

            f_trial = obj_fun(x_trial)
            g_trial_raw = c_obj_fun(x_trial)

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

            grad_f_next = grad_obj_fun(x_next)
            g_next_raw = c_obj_fun(x_next)
            J_next_raw = jac_c_obj_fun(x_next)

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

    return xk, obj_fun(xk), history


def tr_sqp(
    obj_fun,
    grad_obj_fun,
    c_obj_fun,
    jac_c_obj_fun,
    hess_c_obj_fun,
    x0,
    xl,
    xu,
    gl=[0.0, 0.0],
    gu=[np.inf,np.inf],
    hessian_mode="exact",
    hess_obj_fun = None,
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

        fk = obj_fun(xk)
        grad_fk = grad_obj_fun(xk)

        # raw constraints g(x)
        gk_raw = c_obj_fun(xk)      # shape (m,)
        Jg_raw = jac_c_obj_fun(xk)  # shape (m, n)
        Hc_raw = hess_c_obj_fun(xk) # list/array of Hessians H_i

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
            Hk = hess_obj_fun(xk)

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

        f_trial = obj_fun(x_trial)
        g_trial_raw = c_obj_fun(x_trial)

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
                grad_f_next = grad_obj_fun(x_next)
                g_next_raw = c_obj_fun(x_next)
                J_next_raw = jac_c_obj_fun(x_next)

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

    return xk, obj_fun(xk), history
