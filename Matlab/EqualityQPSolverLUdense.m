function [x, lambda] = EqualityQPSolverLUdense(H, g, A, b)

    n = size(H,1);
    m = size(A,2);

    % KKT matrix
    K = [H, -A;
      -A', zeros(m,m)];

    d = -[g; b];

    % Solve using LU factorization
    [L,U,p] = lu(K,"vector");
    z = U \ (L\ d(p) );
    
    x = z(1:n);
    lambda = z(n+1:end);
end


