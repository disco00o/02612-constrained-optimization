function [x, lambda] = EqualityQPSolverLDLdense(H, g, A, b)
    
    n = size(H,1);
    m = size(A,2);

    % Construct KKT System
    K = [H -A
        -A' zeros(m)];

    d = - [g; b];
    
    % Solve using LDL-factorization
    z = zeros(n+m,1);
    [L,D,p] = ldl(K,"lower","vector");
    z(p) = L' \ (D \ (L \ d(p) ) );
    
    x = z(1:n);
    lambda = z(n+1:end);

end