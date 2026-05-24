function [x, lambda] = EqualityQPSolverPlainInverse(H, g, A, b)

    n = size(H,1);
    m = size(A,2);

    % KKT matrix
    K = [H, -A;
      -A', zeros(m,m)];

    d = -[g; b];

   
    z = inv(K) * d;

    x = z(1:n);
    lambda = z(n+1:end);
end
