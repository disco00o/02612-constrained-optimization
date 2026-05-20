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

def convert_QP_to_standard_form(A, b_lower, b_upper, x_lower, x_upper):

    n = len(x_lower)

    # stack the appropriate arrays on top of each other to get the constraints
    # entirely on standard form Ax >= b
    A_new = sparse.vstack((A.T, -A.T, np.eye(n), -np.eye(n)))
    A_new = A_new.tocsr()

    b_new = np.concatenate((b_lower, -b_upper, x_lower, -x_upper))

    return A_new, b_new

def compute_alpha(var, dvar):
        idx = dvar < 0
        if np.any(idx):
            return np.min(-var[idx] / dvar[idx])
        return 1