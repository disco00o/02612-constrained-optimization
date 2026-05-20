import numpy as np
from scipy import sparse
from helper_functions import convert_QP_to_standard_form, compute_alpha
import matplotlib.pyplot as plt


def primal_dual_interior_point(H, g, A, b_lower, b_upper, x_lower, x_upper, x0):
    
    # convert problem to standard form
    A, b = convert_QP_to_standard_form(A, b_lower, b_upper, x_lower, x_upper)

    n = len(x0)
    m = len(b)

    x = x0
    z = np.ones(m)
    s = np.ones(m)

    # compute initial residuals
    rL = H @ x + g - A.T @ z
    rA = s + b - A @ x
    rsz = s*z
    mu = np.sum(rsz) / m

    # track solution path
    xs = []
    rLs = []
    rAs = []
    mus = []
    xs.append(x0)
    rLs.append(np.linalg.norm(rL, ord=np.inf))
    rAs.append(np.linalg.norm(rA, ord=np.inf))
    mus.append(mu)

    max_iter = 5*H.shape[0]
    converged = False

    iter = 0
    while (not converged) and iter < max_iter:
        iter += 1

        # affine step
        Hbar = H + A.T @ sparse.diags(z / s) @ A
        rLbar = rL - A.T @ ((z / s) * (rA - rsz / z))

        # solve for the affine search direction
        x_aff = sparse.linalg.spsolve(Hbar, -rLbar)
        z_aff = -(z / s) * (A @ x_aff) + (z / s) * (rA - rsz / z)
        s_aff = -(rsz / z) - (s / z) * z_aff

        # determine the maximum step size we can take without violating constraints
        alpha_z = compute_alpha(z, z_aff)
        alpha_s = compute_alpha(s, s_aff)
        alpha_aff = np.min([1, alpha_z, alpha_s])

        # compute affine duality gap
        mu_aff = ((z + alpha_aff * z_aff) @ (s + alpha_aff * s_aff)) / m

        # compute centering parameter
        sigma = (mu_aff / mu)**3

        # corrector step
        rszbar = rsz + s_aff*z_aff - sigma*mu*np.ones(len(rsz))

        # update the rhs of the Newton system of equations
        rLbar = rL - A.T @ ((z / s) * (rA - rszbar / z))
        
        # solve the combined system
        dx = sparse.linalg.spsolve(Hbar, -rLbar)

        # flatten the dot product to guarantee a strict 1D array (m,)
        Adx_1D = (A @ dx).flatten()
        dz = -(z / s) * Adx_1D + (z / s) * (rA - rszbar / z).flatten()
        ds = -(rszbar / z) - (s / z) * dz

        # determine maximum step size for the combined direction
        alpha_dz = compute_alpha(z, dz)
        alpha_ds = compute_alpha(s, ds)
        alpha = np.min([1, alpha_dz, alpha_ds])

        eta = 0.995
        alpha_bar = eta * alpha

        # take the step
        x = x + alpha_bar * dx
        z = z + alpha_bar * dz
        s = s + alpha_bar * ds
        xs.append(x)

        # compute residuals
        rL = H @ x + g - A.T @ z
        rA = s + b - A @ x
        rsz = s*z
        mu = np.sum(rsz) / m

        rLnorm = np.linalg.norm(rL, ord=np.inf)
        rAnorm = np.linalg.norm(rA, ord=np.inf)
        rLs.append(rLnorm)
        rAs.append(rAnorm)
        mus.append(mu)

        # check convergence
        converged = (rLnorm < 1e-7) and (rAnorm < 1e-7) and (mu < 1e-7)

    # objective value 
    obj_val = 1/2 * x @ H @ x + g @ x
    
    return x, z, s, obj_val, xs, rLs, rAs, mus


if __name__ == "__main__":

    # load test problem
    H = sparse.load_npz("test_problems/test_data_H_size_100.npz")
    A = sparse.load_npz("test_problems/test_data_A_size_100.npz")
    g, b_lower, b_upper, x_lower, x_upper = np.load("test_problems/test_problem_size_100.npz").values()

    x0 = np.zeros(len(x_lower))

    x, z, s, obj_val, xs, rLs, rAs, mus = primal_dual_interior_point(H, g, A, b_lower, b_upper, x_lower, x_upper, x0)
    # print(f"Minimizer: {x}")
    print(f"Found objective: {obj_val}")

    # plot convergence graph with logarithmic y-axis
    plt.figure(figsize=(10, 6))

    plt.plot(np.arange(len(xs)), rLs, marker='o', linestyle='-', label=r'$\|\|r_L\|\|_{\infty}$')
    plt.plot(np.arange(len(xs)), rAs, marker='o', linestyle='-', label=r'$\|\|r_A\|\|_{\infty}$')
    plt.plot(np.arange(len(xs)), mus, marker='o', linestyle='-', label=r'$\mu$')

    tolerance = 1e-7
    plt.axhline(y=tolerance, color='red', linestyle='--', linewidth=1.5, label='Tolerance ($10^{-7}$)')

    plt.yscale('log')
    plt.xticks(np.arange(len(xs)))

 
    plt.xlabel('Iteration', fontsize=12)
    plt.title('Primal-dual interior-point algorithm convergence', fontsize=14)

    plt.grid(True, which="both", linestyle=':', alpha=1)

    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"figures/interior_point_convergence_size_{H.shape[0]}.png")


