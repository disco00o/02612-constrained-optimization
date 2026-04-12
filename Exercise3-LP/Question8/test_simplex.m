% 1. Generate problem
%n = 100;
n = 10;
density = 0.2; 
beta = 0.5;

%[g, A, bl, bu, xl, xu, x_star] = RandomLP(n, density, beta, 'seed', 42, 'verbose', true);
[g, A, bl, bu, xl, xu, x_star] = RandomLP(n, density, beta, 'seed', 34, 'verbose', true);

% 2. Solve and time
tic;
[x, info, mu, lambda, iter, history, iter1] = solve_lp(g, A, bl, bu, xl, xu);
time = toc;

% 3. Verification
fval = g' * x;
expected_fval = g' * x_star;
primal_error = norm(x - x_star);

% 4. Display results
fprintf('\n===== SOLUTION STATS (solve_lp) =====\n');
fprintf('Status/Info:          %d\n', info); 
fprintf('Found Obj:            %.6f\n', fval);
fprintf('Expected Obj:         %.6f\n', expected_fval);
fprintf('Primal Error:         %.2e\n', primal_error);
fprintf('Total Iterations:     %d\n', iter);
fprintf('CPU time:             %.4f seconds\n', time);

% 5. Feasibility Check
constr_violation = max([0; A'*x - bu; bl - A'*x; xl - x; x - xu]);
fprintf('Max Constraint Viol:  %.2e\n', constr_violation);

% 6. Plotting
if info
    figure('Color', 'w', 'Name', 'Simplex Progress');
    
    % Objective Plot
    subplot(2,1,1);
    plot(1:iter, history.obj, 'LineWidth', 1.5);
    hold on;
    xline(iter1, '--k', 'Phase II', 'LabelVerticalAlignment', 'bottom'); % Vertical line
    grid on; title('Objective Value'); ylabel('g^T x');
    
    % Optimality Violation Plot
    subplot(2,1,2);
    semilogy(1:iter, history.rc + 1e-15, '-r', 'LineWidth', 1.5);
    hold on;
    xline(iter1, '--k'); % Vertical line without label to keep it clean
    grid on; title('Optimality Violation (Reduced Costs < 0)');
    xlabel('Iteration'); ylabel('Norm of Negative \lambda_N');
end