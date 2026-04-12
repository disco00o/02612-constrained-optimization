% 1. Generate problem
%n = 100;
n = 10;
density = 0.2;
beta = 0.5;

%[g, A, bl, bu, xl, xu, x_star] = RandomLP(n, density, beta, 'seed', 42, 'verbose', true);
[g, A, bl, bu, xl, xu, x_star] = RandomLP(n, density, beta, 'seed', 34, 'verbose', true);

% 2. Solve and time
tic;
[x, info, mu, lambda, iter, history] = LPStandardForm(g, A, bl, bu, xl, xu);
time = toc;

% 3. Verification: Compare found solution with the constructed x_star
fval = g' * x;
expected_fval = g' * x_star;
primal_error = norm(x - x_star);

% 4. Display results
fprintf('\n===== SOLUTION STATS (LPStandardForm) =====\n');
fprintf('Status/Info:          %d\n', info); 
fprintf('Found Obj:            %.6f\n', fval);
fprintf('Expected Obj:         %.6f\n', expected_fval);
fprintf('Primal Error:         %.2e\n', primal_error);
fprintf('Iterations:           %d\n', iter);
fprintf('CPU time:             %.4f seconds\n', time);

% 5. Feasibility Check
% Ensuring the custom solver respected the bounds and constraints
constr_violation = max([0; A'*x - bu; bl - A'*x; xl - x; x - xu]);
fprintf('Max Constraint Viol:  %.2e\n', constr_violation);


% 6. Convergence plot
if info
    figure('Color', 'w', 'Name', 'Convergence Analysis');
    iters = 1:iter;
    
    semilogy(iters, history.rA, '-o', 'Color', [0 0.447 0.741], 'LineWidth', 1.5, 'MarkerSize', 4);
    hold on;
    semilogy(iters, history.rL, '-s', 'Color', [0.85 0.325 0.098], 'LineWidth', 1.5, 'MarkerSize', 4);
    semilogy(iters, history.s, '-d', 'Color', [0.466 0.674 0.188], 'LineWidth', 1.5, 'MarkerSize', 4);
    
    % Add tolerance threshold
    line([1 iter], [1e-9 1e-9], 'Color', 'r', 'LineStyle', '--', 'HandleVisibility', 'off');
    text(1, 2e-9, 'Tolerance (10^{-9})', 'Color', 'r', 'FontSize', 9);
    
    grid on;
    xlabel('Iteration Number');
    ylabel('Residual Norm (log scale)');
    title(['Algorithm Convergence Profile (n = ', num2str(length(g)), ')']);
    legend('Primal Feasibility ||Ax-b||_\infty', ...
           'Dual Feasibility ||r_L||_\infty', ...
           'Duality Gap (s)', ...
           'Location', 'northeast');
else
    warning('Solver did not converge. Plot skipped.');
end