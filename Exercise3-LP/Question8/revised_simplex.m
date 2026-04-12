function [x, lambda, mu, B, N, iter, hist] = revised_simplex(g, A, b, B, N)
    max_iter = 1000; iter = 0; STOP = false;
    hist.obj = []; hist.rc = []; 
    
    while ~STOP && iter < max_iter
        iter = iter + 1;
        
        % --- 1. Factorize Basis and Compute Duals ---
        B_mat = A(:, B); 
        N_mat = A(:, N);
        mu = B_mat' \ g(B); 
        lambda_N = g(N) - N_mat' * mu;
        
        % --- 2. Track Convergence Progress ---
        hist.obj(iter,1) = g(B)' * (B_mat \ b);
        hist.rc(iter,1)  = norm(lambda_N(lambda_N < 0)); 
        
        % --- 3. Optimality Test ---
        if all(lambda_N >= -1e-10)
            STOP = true;
        else
            % --- 4. Select Entering Variable ---
            [~, s] = min(lambda_N); 
            i_s = N(s);
            
            % --- 5. Minimum Ratio Test ---
            h = B_mat \ A(:, i_s);
            x_B = B_mat \ b;
            
            candidates = find(h > 1e-10);
            if isempty(candidates), error('Problem is Unbounded'); end
            
            ratios = x_B(candidates) ./ h(candidates);
            alpha = min(ratios);
            
            % Identify leaving variable
            J = candidates(abs(ratios - alpha) < 1e-10);
            j = J(1); 
            i_j = B(j); 
            
            % --- 6. Update Basis/Non-Basis Sets ---
            B(j) = i_s; 
            N(s) = i_j; 
        end
    end
    
    % --- 7. Final Output Construction ---
    B_mat = A(:, B);
    x = zeros(size(g)); 
    x(B) = B_mat \ b;
    lambda = zeros(size(g)); 
    lambda(N) = lambda_N;
end