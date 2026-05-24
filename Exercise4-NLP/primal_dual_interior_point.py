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
        H = sparse.csr_matrix(H)
        A = sparse.csr_matrix(A)
        D = sparse.diags(z / s)

        Hbar = H + A.T @ D @ A
        Hbar = Hbar + 1e-8 * sparse.eye(Hbar.shape[0], format="csc")

        # Ensure CSC format for spsolve
        Hbar = Hbar.tocsc()

        # Hbar = H + A.T @ sparse.diags(z / s) @ A

        # # Make sure Hbar is positive definite by adding a small multiple of the identity matrix
        # Hbar = Hbar + 1e-8 * sparse.eye(Hbar.shape[0])

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


