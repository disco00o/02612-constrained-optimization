function [x, info, mu, lambda, iter, history, iter1] = solve_lp(g, A_in, bl, bu, xl, xu)
    [n, m] = size(A_in);
    
    % --- Conversion to Standard Form ---
    I_m = eye(m); I_n = eye(n);
    A_bar = [A_in', -I_m,  zeros(m), zeros(m, n);
             A_in',  zeros(m), I_m,  zeros(m, n);
             I_n,    zeros(n, m), zeros(n, m), I_n];
    b_bar = [bl - A_in'*xl; bu - A_in'*xl; xu - xl];
    g_bar = [g; zeros(2*m + n, 1)];

    % --- Phase 1 ---
    [m_bar, n_bar] = size(A_bar);
    g_p1 = [zeros(n_bar, 1); ones(2*m_bar, 1)];
    A_p1 = [A_bar, eye(m_bar), -eye(m_bar)];
    
    % Initialization
    B = zeros(1, m_bar);
    for i = 1:m_bar
        if b_bar(i) >= 0, B(i) = n_bar + i; else, B(i) = n_bar + m_bar + i; end
    end
    N = setdiff(1:(n_bar + 2*m_bar), B);

    [X_p1, ~, ~, B_f1, ~, iter1, hist1] = revised_simplex(g_p1, A_p1, b_bar, B, N);
    
    % Feasibility Check
    if sum(X_p1(n_bar+1:end)) > 1e-7
        info = 0; x = zeros(n,1); mu = []; lambda = []; iter = iter1; history = hist1;
        return;
    end

    % --- Phase 2 ---
    B_p2 = B_f1;
    for i = 1:m_bar
        if B_p2(i) > n_bar
            orig_N = find(N <= n_bar);
            B_p2(i) = N(orig_N(1));
            N(orig_N(1)) = [];
        end
    end
    N_p2 = setdiff(1:n_bar, B_p2);

    [x_aug, lam_aug, mu, B_f2, N_f2, iter2, hist2] = revised_simplex(g_bar, A_bar, b_bar, B_p2, N_p2);

    % Final Mapping
    x = x_aug(1:n) + xl;
    lambda = lam_aug; 
    iter = iter1 + iter2;
    info = 1;
    
    % Merge Histories
    history.obj = [hist1.obj; hist2.obj];
    history.rc  = [hist1.rc; hist2.rc];
end