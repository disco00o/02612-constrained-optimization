import numpy as np
from scipy.sparse import random, eye, issparse, bmat, diags, triu, csc_matrix
from scipy.stats import norm, uniform
from scipy.linalg import ldl, solve, solve_triangular
import qdldl # Required for sparse LDL factorization
# You may need to do the cmds below for the kernel to not crash:
# pip uninstall qdldl
# pip install qdldl --no-binary qdldl

"""
# Exercise 1.3 - Random EQP
"""

def RandomEQP(n, alpha, density, beta, flag='dense', seed=None):

    # Initialize Random Number generator with seed for reproducability
    rng = np.random.default_rng(seed)

    n = int(n)
    # Get Columns 
    m = int(np.round(beta * n))
    
    if m > 0:
        # Create A matrix (Note, the constrained reads A'*x = b)
        # -> Drawing from a normal distribution
        # -> It will be generated in sparse format: 'csc' - compressed sparse column
        A = random(n, m, density=density, format='csc', 
                        data_rvs=lambda size: norm.rvs(size=size, random_state=rng),
                        random_state=rng)
        b = uniform.rvs(loc=-1, scale=2, size=(m, 1))
            
    else:
        A = csc_matrix((n, 0)) # Empty Sparse-Matrix
        b = np.array([]).reshape(0, 1) # Empty vector


    M = random(n, n, density=density, format='csc', 
                    data_rvs=lambda size: norm.rvs(size=size, random_state=rng),
                    random_state=rng)
    
    H = M @ M.T + alpha * eye(n, format='csc') 

    # Create g and b
    # -> Drawing from a uniform distribution in the interval -1 to 1
    g = uniform.rvs(loc=-1, scale=2, size=(n, 1))

    # Convert to dense 'normal' matrix, if desired
    match flag:
        case 'dense':
            H = H.toarray()
            A = A.toarray()
            assert np.all(np.linalg.eigvals(H) > 0) 
        case 'sparse':
            H_d = H.toarray()
            assert np.all(np.linalg.eigvals(H_d) > 0), "Non convex problem generated!"
            pass
        case 'sparse&dense':
            H_d = H.toarray()
            assert np.all(np.linalg.eigvals(H_d) > 0), "Non convex problem generated!"
            A_d = A.toarray()
            return H, H_d, g, A, A_d, b
        case _:
            print("Invalid Flag!")

    "Non convex problem generated!"
    
    return H, g, A, b

"""
# Exercise 1.4: Solvers based on LDL-factorization
"""

def EqualityQPSolver(H, g, A, b, solver='LDLsparse', verbose=True):
    
    n = H.shape[0]
    m = A.shape[1]

    match solver:
        case 'LDLsparse':
            assert (issparse(H) and issparse(A)), "Matrices must be sparse!!!"
            # Construct the sparse KKT matrix
            #K = bmat([[H, -A], [-A.T, None]], format='csc')
            Z = csc_matrix((m, m))

            K = bmat([[H, A.T],
                    [A, Z]], format='csc')

            # Enforce float64
            K = K.astype(np.float64)

            # Enforce structural symmetry: keep only upper triangle
            K = triu(K, format='csc')
            K.eliminate_zeros()

            factorization = qdldl.Solver(K)

            rhs = -np.vstack([g, b]).flatten()
            # L = factorization.L
            # D = diags(factorization.D)
            # z(p) = L’ \ ( D \ ( L \ d(p) ) );
            # -> Use qdldl library to solve sparse system
            sol = factorization.solve(rhs)
            
            x = sol[:n]
            lamb = sol[n:]

            
        
        case 'LDLdense':
            
            K = np.block([[H, -A], [-A.T, np.zeros((m, m))]])
            d = -np.concatenate([g.flatten(), b.flatten()])
            
            # LDL factorization
            L, D, p = ldl(K, lower=True)
            # z(p) = L’ \ ( D \ ( L \ d(p) ) );
            # In Matlab this system is solved automatically by using the \,
            # but in Python we need to do it manually
            # Start by solving L \ d(p)
            d_p = d[p] # Permutate d vector
            tmp = solve_triangular(L, d_p, lower=True, unit_diagonal=True)

            # Solve  D \ ( L \ d(p) ) :  D \ (tmp)
            # -> Using the generic solve, because no matrix is guranteed to be
            #    triangular
            tmp = solve(D, tmp)

            # Solve L’ \ ( D \ ( L \ d(p) ) ) : L’ \ tmp )
            # -> Solve using triangular solver, as L' will be an upper triangular matrix

            z_p = solve_triangular(L.T, tmp, lower=False, unit_diagonal=True)

            # Undo the permutation
            z = np.zeros_like(z_p)
            z[p] = z_p

            x = z[:n]
            lamb = z[n:]
        
        case 'plaininverse':
            K = np.block([[H, -A], [-A.T, np.zeros((A.shape[1], A.shape[1]))]])
            rhs = -np.concatenate([g.flatten(), b.flatten()])
            
            # Just take the inverse
            z = np.linalg.inv(K) @ rhs
            
            x = z[:n]
            lamb = z[n:]

        case _: 
            assert f"{solver} is not a valid solver!"      
        
    
    if verbose:
        print(f"Solver: {solver}, found x: {x}, lambda: {lamb}")

    return x, lamb
