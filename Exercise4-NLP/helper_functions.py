import numpy as np
from scipy import sparse

def equalityQPsolver(H, g, A, b):

    n = H.shape[0]
    m = A.shape[0]

    upper = sparse.hstack([H, A.T])
    zero_mat = np.zeros((m,m))
    lower = sparse.hstack([A,zero_mat])

    KKT = sparse.vstack([upper,lower])
    KKT = KKT.tocsr()
    # regularization = sparse.eye(KKT.shape[0]) * 1e-8      # handled in the algorithm code for now
    # KKT = KKT + regularization

    rhs = np.hstack([-g,b])

    sol = sparse.linalg.spsolve(KKT, rhs)

    x = sol[:n]
    lambda_ = sol[n:]

    return x, lambda_

# def convert_QP_to_standard_form(A, b_lower, b_upper, x_lower, x_upper):

#     n = len(x_lower)

#     # stack the appropriate arrays on top of each other to get the constraints
#     # entirely on standard form Ax >= b
#     A_new = sparse.vstack((A.T, -A.T, np.eye(n), -np.eye(n)))
#     A_new = A_new.tocsr()

#     b_new = np.concatenate((b_lower, -b_upper, x_lower, -x_upper))

#     return A_new, b_new

from scipy import sparse
import numpy as np

from scipy import sparse
import numpy as np

# def convert_QP_to_standard_form(A, b_lower, b_upper, x_lower, x_upper):
#     n = len(x_lower)

#     A = sparse.csr_matrix(A)
#     I = sparse.eye(n, format="csr")

#     b_lower = np.asarray(b_lower).ravel()
#     b_upper = np.asarray(b_upper).ravel()

#     A_new = sparse.vstack((A.T, -A.T, I, -I), format="csr")
#     b_new = np.concatenate((b_lower, -b_upper, x_lower, -x_upper))

#     return A_new, b_new

def convert_QP_to_standard_form(A, b_lower, b_upper, x_lower, x_upper):
    n = len(x_lower)
    A = sparse.csr_matrix(A)
    I = sparse.eye(n, format="csr")

    b_lower = np.asarray(b_lower).ravel()
    b_upper = np.asarray(b_upper).ravel()
    x_lower = np.asarray(x_lower).ravel()
    x_upper = np.asarray(x_upper).ravel()

    # Base blocks (always included)
    # Note: A is assumed to be shape (n, m), so A.T is shape (m, n)
    A_blocks = [A.T, I]
    b_blocks = [b_lower, x_lower]

    # Vectorized filter for finite upper constraints
    finite_b_upper = b_upper < 1e10
    if np.any(finite_b_upper):
        A_blocks.append(-A.T[finite_b_upper, :])
        b_blocks.append(-b_upper[finite_b_upper])

    # Vectorized filter for finite upper variable bounds
    finite_x_upper = x_upper < 1e10
    if np.any(finite_x_upper):
        A_blocks.append(-I[finite_x_upper, :])
        b_blocks.append(-x_upper[finite_x_upper])

    # Assemble into standard form
    A_new = sparse.vstack(A_blocks, format="csr")
    b_new = np.concatenate(b_blocks)

    return A_new, b_new



def compute_alpha(var, dvar):
        idx = dvar < 0
        if np.any(idx):
            return np.min(-var[idx] / dvar[idx])
        return 1