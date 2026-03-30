import numpy as np
from scipy.sparse import random, eye, issparse
from scipy.stats import norm, uniform

"""
# Exercise 1.3 - Random EQP
"""

def RandomEQP(n, alpha, density, beta, flag='dense'):

    # Get Columns 
    m = int(np.round(beta * n))
    
    # Create A matrix (Note, the constrained reads A'*x = b)
    # -> Drawing from a normal distribution
    # -> It will be generated in sparse format: 'csc' - compressed sparse column
    A = random(n, m, density=density, format='csc', data_rvs=norm.rvs)

    M = random(n, n, density=density, format='csc', data_rvs=norm.rvs)
    H = M @ M.T + alpha * eye(n, format='csc') 

    # Create g and b
    # -> Drawing from a uniform distribution in the interval -1 to 1
    g = uniform.rvs(loc=-1, scale=2, size=(n, 1))
    b = uniform.rvs(loc=-1, scale=2, size=(m, 1))

    # Convert to dense 'normal' matrix, if desired
    if flag == 'dense':
        H = H.toarray()
        A = A.toarray()
    elif flag == 'sparse':
        pass
    else:
        print("Invalid Flag!")

    return H, g, A, b

"""
# Exercise 1.4: Solvers based on LDL-factorization
"""

def EqualityQPSolver(H, g, A, b, solver='LDLsparse'):
    match solver:
        case 'LDLsparse':
            assert (issparse(H) and issparse(A))
            "Matrices must be sparse!!!"
            # Construct the KKT system
            KKT_matrix = np.block([[H, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
            



            return x, lamb
        
        case 'LDLdense':
        
            return x, lamb
        
        case 'LU':
        
            return x, lamb
        
        case 'plaininverse':

            return x, lamb
        
        case _: 
            assert f"{solver} is not a valid solver!"

