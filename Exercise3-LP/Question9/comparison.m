%% Performance Comparison of LP Solvers
% Solvers: linprog (MathWorks), LPStandardForm (Interior Point), solve_lp (Simplex)

clear; clc; close all;

% Configuration
n_values = [10, 20, 50, 100, 150, 200, 300]; % Test sizes
density = 0.2;
beta = 0.5;
num_solvers = 4;
labels = {'linprog (Dual-Simplex)', 'linprog (Interior-Point)', 'Primal-Dual Interior-Point', 'Revised Simplex'};

% Initialize data storage
mse_results = zeros(length(n_values), num_solvers);
iter_results = zeros(length(n_values), num_solvers);
time_results = zeros(length(n_values), num_solvers);

for i = 1:length(n_values)
    n = n_values(i);
    fprintf('Testing n = %d...\n', n);
    
    % 1. Generate problem
    [g, A, bl, bu, xl, xu, x_star] = RandomLP(n, density, beta, 'seed', 34, 'verbose', false);
    expected_fval = g' * x_star;
    
    % Convert for linprog
    Aineq = [A'; -A'];
    bineq = [bu; -bl];
    
    %% --- Solver 1: linprog (Dual-Simplex) ---
    opt1 = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'off');
    tic;
    [x1, fval1, ~, out1] = linprog(g, Aineq, bineq, [], [], xl, xu, opt1);
    time_results(i, 1) = toc;
    mse_results(i, 1) = (fval1 - expected_fval)^2;
    iter_results(i, 1) = out1.iterations;
    
    %% --- Solver 2: linprog (Interior-Point) ---
    opt2 = optimoptions('linprog', 'Algorithm', 'interior-point', 'Display', 'off');
    tic;
    [x2, fval2, ~, out2] = linprog(g, Aineq, bineq, [], [], xl, xu, opt2);
    time_results(i, 2) = toc;
    mse_results(i, 2) = (fval2 - expected_fval)^2;
    iter_results(i, 2) = out2.iterations;
    
    %% --- Solver 3: LPStandardForm (Custom Interior Point) ---
    tic;
    [x3, info3, ~, ~, iter3, ~] = LPStandardForm(g, A, bl, bu, xl, xu);
    time_results(i, 3) = toc;
    fval3 = g' * x3;
    mse_results(i, 3) = (fval3 - expected_fval)^2;
    iter_results(i, 3) = iter3;
    
    %% --- Solver 4: solve_lp (Custom Simplex) ---
    tic;
    [x4, info4, ~, ~, iter4, ~, ~] = solve_lp(g, A, bl, bu, xl, xu);
    time_results(i, 4) = toc;
    fval4 = g' * x4;
    mse_results(i, 4) = (fval4 - expected_fval)^2;
    iter_results(i, 4) = iter4;
end

%% Plotting Results
colors = [0 0.447 0.741; 0.85 0.325 0.098; 0.929 0.694 0.125; 0.494 0.184 0.556];
markers = {'-o', '--s', '-^', '-d'};

figure('Name', 'LP Solver Comparison', 'Color', 'w', 'Position', [100, 100, 1000, 900]);

% 1. Mean Squared Error (MSE)
subplot(3, 1, 1);
for s = 1:num_solvers
    semilogy(n_values, mse_results(:, s), markers{s}, 'Color', colors(s,:), 'LineWidth', 1.5); hold on;
end
grid on; title('Accuracy: Mean Squared Error (Objective Value)');
ylabel('MSE (log scale)'); legend(labels, 'Location', 'eastoutside');

% 2. Iterations
subplot(3, 1, 2);
for s = 1:num_solvers
    plot(n_values, iter_results(:, s), markers{s}, 'Color', colors(s,:), 'LineWidth', 1.5); hold on;
end
grid on; title('Efficiency: Iteration Count');
ylabel('Iterations'); legend(labels, 'Location', 'eastoutside');

% 3. CPU Time
subplot(3, 1, 3);
for s = 1:num_solvers
    semilogy(n_values, time_results(:, s), markers{s}, 'Color', colors(s,:), 'LineWidth', 1.5); hold on;
end
grid on; title('Performance: CPU Execution Time');
ylabel('Seconds (log scale)'); xlabel('Problem Dimension (n)');
legend(labels, 'Location', 'eastoutside');

fprintf('\nAnalysis complete.\n');