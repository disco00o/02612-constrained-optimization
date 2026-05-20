import numpy as np
from scipy import sparse, optimize
from helper_functions import equalityQPsolver, convert_QP_to_standard_form


def find_feasible_initial_point(A, b):
    n, m = A.shape

    # define the LP we want to solve to find a feasible x
    c = np.append(np.zeros(m), 1)
    A_new = sparse.hstack((A, np.ones((n, 1))))

    # scipys linprog optimize.linprog takes constraints of the form Ax <= b
    # so we multiply the constraints by -1 to get the correct form
    A_new = -A_new
    b_new = -b

    # every x1, x2,...xn should be unbounded in this LP, whereas s should be non-negative
    bounds = [(None, None)] * m + [(0, None)]

    # solve the LP
    sol = optimize.linprog(c, A_ub=A_new, b_ub=b_new, bounds=bounds)

    if sol.success:
        return sol.x[:-1]
    else:
        print("Problem is infeasible")


def primal_active_set(H, g, A, b_lower, b_upper, x_lower, x_upper):
    
    # convert problem to standard form
    A, b = convert_QP_to_standard_form(A, b_lower, b_upper, x_lower, x_upper)

    # solve a LP to find feasible initial point
    x0 = find_feasible_initial_point(A, b)

    # track solution path
    xs = []
    xs.append(x0)
    x = x0
    max_iter = 5*H.shape[0]
    tol = 10e-4

    # initialize working set (true means constraint is active)
    W = np.zeros(A.shape[0], dtype=bool)

    # find initial active set
    for i in range(A.shape[0]):
        if np.isclose(A[i,:] @ x, b[i]):
            W[i] = True
    
    for _ in range(max_iter):
        
        # find search direction by solving equality constrained QP for current active constraints
        p, lambda_p = equalityQPsolver(H, (H@x+g), A[W], np.zeros(A[W].shape[0]))
        
        lambda_p *= -1

        # check if p is zero (i.e. we cannot improve with the current working set)
        if np.linalg.norm(p, ord=2) < tol:
            # if all langrange multipliers are non-negative KKT conditions are satisfied and we have 
            # a global minimum
            if np.all(lambda_p >= 0):
                lambda_ = np.zeros(len(W))
                lambda_[W] = lambda_p

                # objective value
                obj_val = 1/2 * x @ H @ x + g @ x

                return x, lambda_, obj_val, xs
            else:
                # if there are negative multipliers, current objective can be improved
                # find the index of the first negative multiplier
                idx = np.argwhere(lambda_p < 0)[0][0]
                idx2 = np.where(W)[0]

                # remove index j from active set
                W[idx2[idx]] = False
        
        # if p is non-zero we want to step in direction p
        else:
            # find inactive constraints
            alpha_W = ~W
            alpha_W[A @ p >= 0] = False

            # compute step sizes in direction p to reach inactive constraints
            inactive_distances = np.array((b[alpha_W] - A[alpha_W] @ x) / (A[alpha_W] @ p))

            # set step size to smallest value
            if len(inactive_distances) > 0:
                alpha = np.min(inactive_distances)
            else:
                alpha = 1

            if alpha < 1:
                
                # take step to reach new constraint
                x = x + alpha * p
                xs.append(x.copy())
                
                # identify the constraint we reached
                idx3 = np.argmin(inactive_distances)
                valid_indices = np.where(alpha_W)[0] 

                # add index to the active set
                W[valid_indices[idx3]] = True
            
            # if alpha >= 1, we take a full step (i.e. alpha = 1)
            else:
                x = x + p
                xs.append(x.copy())


if __name__ == "__main__":

    # load test problem
    H = sparse.load_npz("test_problems/test_data_H_size_100.npz")
    A = sparse.load_npz("test_problems/test_data_A_size_100.npz")
    g, b_lower, b_upper, x_lower, x_upper = np.load("test_problems/test_problem_size_100.npz").values()

    x, lambda_, obj_val, xs = primal_active_set(H, g, A, b_lower, b_upper, x_lower, x_upper)
    # print(f"Minimizer: {x}")
    print(f"Found objective: {obj_val}")
