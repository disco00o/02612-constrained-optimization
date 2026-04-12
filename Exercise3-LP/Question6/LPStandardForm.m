function [x_final, info, mu, lambda, iter, history] = LPStandardForm(g, A, bl, bu, xl, xu)
    % --- 1. PRE-PROCESSING ---
    [n, m] = size(A); % n is length of g, m is number of constraints (rows of A')
    Im = eye(m); In = eye(n);
    
    % Build the matrix bar_A based on:
    % [ A' -I  0  0 ]
    % [ A'  0  I  0 ]
    % [ I   0  0  I ]
    bar_A = [A', -Im,         zeros(m,m), zeros(m,n);
             A',  zeros(m,m),  Im,         zeros(m,n);
             In,  zeros(n,m),  zeros(n,m),  In];
             
    % Build Super Vector bar_b
    % [ bl - A'*xl ]
    % [ bu - A'*xl ]
    % [ xu - xl    ]
    bar_b = [bl - A'*xl; 
             bu - A'*xl; 
             xu - xl];
    
    % Build Super Vector bar_g
    % [ g; 0; 0; 0 ]
    bar_g = [g; zeros(2*m + n, 1)];
    
    % --- Initial Guess for bar_x ---
    % bar_x = [x_hat; s1; s2; s3]
    % We initialize x_hat at a small positive value (since x_hat >= 0)
    x_hat_init = ones(n, 1) * 0.1; 
    
    % Calculate slacks based on the equality bar_A * bar_x = bar_b
    % s1 = A'*x_hat - (bl - A'*xl)
    % s2 = (bu - A'*xl) - A'*x_hat
    % s3 = (xu - xl) - x_hat
    s1 = A'*x_hat_init - (bl - A'*xl);
    s2 = (bu - A'*xl) - A'*x_hat_init;
    s3 = (xu - xl) - x_hat_init;
    
    % Ensure initial slacks are strictly positive for the Interior Point Method
    eps0 = 1.0;
    s1 = max(s1, eps0);
    s2 = max(s2, eps0);
    s3 = max(s3, eps0);
    
    x0 = [x_hat_init; s1; s2; s3];
    
    % --- 2. CALL THE SOLVER ---
    [bar_x, info, mu, lambda, iter, history] = LPippd(bar_g, bar_A, bar_b, x0);

    % --- 3. POST-PROCESSING ---
    % Extract x_hat (first n elements) and shift back: x = x_hat + xl
    if info
        x_hat_final = bar_x(1:n);
        x_final = x_hat_final + xl;
    else
        x_final = [];
    end
end