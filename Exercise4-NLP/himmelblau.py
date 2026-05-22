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

def hess_himmelblau(x):
    x1, x2 = x

    return np.array([
        [12 * x1**2 + 4 * x2 - 42,
         4 * x1 + 4 * x2],

        [4 * x1 + 4 * x2,
         4 * x1 + 12 * x2**2 - 26]
    ])




def solve_qp_subproblem(
    xk,
    Hk,
    grad_fk,
    xl,
    xu,
    gk,
    Jgk
):
    n = len(xk)

    def phi(p):
        return 0.5 * p @ Hk @ p + grad_fk @ p

    def grad_phi(p):
        return Hk @ p + grad_fk

    # bounds on p from box constraints
    p_lower = xl - xk
    p_upper = xu - xk
    bounds = Bounds(p_lower, p_upper)

    # inequality constraints: c_i(xk) + a_i^T p >= 0
    cons = []
    m = len(gk)
    for i in range(m):
        a_i = Jgk[i]

        def fun_factory(c_i, a_i):
            return lambda p, c_i=c_i, a_i=a_i: c_i + a_i @ p

        def jac_factory(a_i):
            return lambda p, a_i=a_i: a_i

        cons.append({
            "type": "ineq",
            "fun": fun_factory(gk[i], a_i),
            "jac": jac_factory(a_i)
        })

    # -------------------------
    # 1) Solve QP using SLSQP
    # -------------------------
    res = minimize(
        phi,
        x0=np.zeros(n),
        jac=grad_phi,
        bounds=bounds,
        constraints=cons,
        method="SLSQP",
    )

    p_slsqp = res.x if res.success else -np.linalg.solve(Hk, grad_fk)

    # -------------------------
    # 2) Solve the SAME QP using interior-point solver
    # -------------------------

    # Direct formulation: b_lower <= J_g @ p <= b_upper
    A = sparse.csr_matrix(Jgk.T)      # (n, m)

    g_l = 0.0
    g_u = 1e20   # conceptually; use large finite in code

    b_lower = g_l - gk                # = -gk
    b_upper = g_u - gk                # stand-in for g_u - gk

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

    # --- decode multipliers ---
    m = len(gk)
    n = len(xk)

    idx_g  = slice(0, m)
    idx_xl = slice(m, m+n)
    idx_xu = slice(m+n, m+2*n)

    lambda_k = z_ipm[idx_g]
    mu_l_k   = z_ipm[idx_xl]
    mu_u_k   = z_ipm[idx_xu]

    print("\n--- QP comparison ---")
    print("SLSQP step:", p_slsqp)
    print("IPM step:  ", pk)
    print("---------------------\n")

    return pk, lambda_k, mu_l_k, mu_u_k


def constraint_violation(g):
    return np.sum(np.maximum(0.0, -g))

def merit_function(f, g, rho):
    return f + rho * constraint_violation(g)



def line_search_sqp(
    x0,
    xl,
    xu,
    hessian_mode="exact",
    max_iter=100,
    tol=1e-8
):

    history = {"f":[],"x": []}


    xk = np.asarray(x0, dtype=float)
    n = len(xk)

    Bk = np.eye(n)

    

    for k in range(max_iter):

        fk = himmelblau(xk)

        history["x"].append(xk.copy())


        grad_fk = grad_himmelblau(xk)
        ck = c_himmelblau(xk)
        Jk = jac_c_himmelblau(xk)

        history["f"].append(fk)

        # stationarity (unconstrained) check
        if np.linalg.norm(grad_fk) < tol:
            break

        # Hessian / quasi-Newton
        if hessian_mode == "exact":
            Hk = hess_himmelblau(xk)
        else:
            Hk = Bk

        eig_min = np.min(np.linalg.eigvalsh(Hk))
        if eig_min <= 1e-8:
            Hk += (abs(eig_min) + 1e-4) * np.eye(n)

        # --- QP subproblem with constraints ---
        
        pk, lambda_k, mu_l_k, mu_u_k = solve_qp_subproblem(
                                                            xk=xk,
                                                            Hk=Hk,
                                                            grad_fk=grad_fk,
                                                            xl=xl,
                                                            xu=xu,
                                                            gk=ck,
                                                            Jgk=Jk
                                                        )


        # --- backtracking line search (ℓ1 merit) ---
        rho = 10.0
        alpha = 1.0
        rho_backtrack = 0.5
        c1 = 1e-4

        phi_k = merit_function(fk, ck, rho)

        while alpha > 1e-12:
            x_trial = xk + alpha * pk
            x_trial = np.minimum(np.maximum(x_trial, xl), xu)

            f_trial = himmelblau(x_trial)
            g_trial = c_himmelblau(x_trial)
            phi_trial = merit_function(f_trial, g_trial, rho)

            if phi_trial <= phi_k + c1 * alpha * grad_fk @ pk:
                break

            alpha *= rho_backtrack

        x_next = x_trial

        # BFGS update
        grad_f_next = grad_himmelblau(x_next)
        J_next      = jac_c_himmelblau(x_next)
        lambda_next = lambda_k
        mu_l_next   = mu_l_k
        mu_u_next   = mu_u_k


        if hessian_mode == "bfgs":
            # Lagrangian gradients
            grad_Lk = grad_fk + Jk.T @ lambda_k + mu_l_k - mu_u_k
            grad_L_next = grad_f_next + J_next.T @ lambda_next + mu_l_next - mu_u_next

            s = x_next - xk
            y = grad_L_next - grad_Lk

            Bk = damped_bfgs_update(Bk, s, y)



        if np.linalg.norm(x_next - xk) < tol:
            xk = x_next
            break

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
    # ∂c1/∂x = [2(x1+2), -1]
    # ∂c2/∂x = [-4, 10]
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



if __name__ == "__main__":
    xl = np.array([-5.0, -5.0])
    xu = np.array([5.0, 5.0])
    x0 = np.array([-5, 0])

    x_exact, f_exact, hist_exact = line_search_sqp(
    x0=x0, xl=xl, xu=xu, hessian_mode="exact"
    )

    x_bfgs, f_bfgs, hist_bfgs = line_search_sqp(
        x0=x0, xl=xl, xu=xu, hessian_mode="bfgs"
    )

    # Plot
    plot_convergence_compare(hist_exact, hist_bfgs)
    plot_paths(hist_exact, hist_bfgs)

    print("Exact SQP optimum x:", x_exact)
    print("Exact SQP optimum f:", f_exact)
    print("BFGS SQP optimum x:", x_bfgs)
    print("BFGS SQP optimum f:", f_bfgs)

    # quick contour + constraints
    X = np.linspace(-5, 5, 400)
    Y = np.linspace(-5, 5, 400)
    XX, YY = np.meshgrid(X, Y)
    ZZ = (XX**2 + YY - 11)**2 + (XX + YY**2 - 7)**2

    c1 = (XX + 2)**2 - YY
    c2 = -4*XX + 10*YY

    plt.figure(figsize=(8, 6))
    cs = plt.contour(XX, YY, ZZ, levels=40)
    plt.clabel(cs, inline=1, fontsize=8)

    # constraint boundaries
    plt.contour(XX, YY, c1, levels=[0], colors="red")
    plt.contour(XX, YY, c2, levels=[0], colors="blue")

    # feasible shading
    feasible = (c1 >= 0) & (c2 >= 0)
    plt.imshow(
        feasible.astype(int),
        extent=[-5, 5, -5, 5],
        origin="lower",
        alpha=0.2,
        cmap="Greens"
    )

    plt.plot(x_exact[0], x_exact[1], "ro", label="SQP optimum")
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("SQP on constrained Himmelblau")
    plt.show()

