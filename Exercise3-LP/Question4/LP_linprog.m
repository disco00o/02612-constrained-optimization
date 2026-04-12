% 1. Generate problem
%n = 100;
n = 10;
density = 0.2;
beta = 0.5;

% Note: We capture x_star to verify if the solver finds the intended global minimum
%[g, A, bl, bu, xl, xu, x_star] = RandomLP(n, density, beta, 'seed', 42, 'verbose', true);
[g, A, bl, bu, xl, xu, x_star] = RandomLP(n, density, beta, 'seed', 34, 'verbose', true);

% 2. Convert constraints to linprog format
% A'x <= bu  AND  A'x >= bl  =>  [A'; -A'] * x <= [bu; -bl]
Aineq = [A'; -A'];
bineq = [bu; -bl];

f = g;
lb = xl;
ub = xu;

% 3. Solver options
options = optimoptions('linprog', ...
    'Algorithm', 'dual-simplex', ...   
    'Display', 'iter');

% 4. Solve and time
tic;
[x, fval, exitflag, output] = linprog(f, Aineq, bineq, [], [], lb, ub, options);
cpu_time = toc;

% 5. Verification: Compare found solution with the constructed x_star
primal_error = norm(x - x_star);
expected_fval = g' * x_star;

% 6. Display results
fprintf('\n===== SOLUTION STATS (linprog) =====\n');
fprintf('Exit flag:            %d\n', exitflag);
fprintf('Found Obj:            %f\n', fval);
fprintf('Expected Obj:         %f\n', expected_fval);
fprintf('Primal Error:         %.2e\n', primal_error);
fprintf('Iterations:           %d\n', output.iterations);
fprintf('CPU time:             %.4f seconds\n', cpu_time);
fprintf('Algorithm:            %s\n', output.algorithm);