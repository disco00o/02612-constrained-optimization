import numpy as np
from scipy import sparse

def generate_test_problem(n, alpha=0.1, density=0.3):

    m = 5*n

    # generate sparse PSD matrix H
    U = sparse.random(n,n,density)
    H = U @ U.T + alpha * sparse.identity(n)

    # sample standard normal distribution
    g = np.random.randn(n)
    
    # generate sparse constraint matrix A
    A = sparse.random(n,m,density)

    x_lower = -np.ones(n)
    x_upper = np.ones(n)

    b_lower = -np.random.rand(m)
    b_upper = np.random.rand(m)

    return H, g, A, b_lower, b_upper, x_lower, x_upper

if __name__ == "__main__":
    n = 100
    H, g, A, b_lower, b_upper, x_lower, x_upper = generate_test_problem(n)

    # save the generated test problem
    np.savez(f"test_problems/test_problem_size_{n}.npz", g, b_lower, b_upper, x_lower, x_upper)
    sparse.save_npz(f"test_problems/test_data_H_size_{n}.npz", H)
    sparse.save_npz(f"test_problems/test_data_A_size_{n}.npz", A)

    # print(type(H))
    # print(type(A))
    # print(H)



