%% Benchmark on week 5 exercise problem
close all;
ns = 1000:1000:4000;          % problem sizes         
comp_plain_inv = true;
comp_LU = true;

t_denseLDL  = zeros(length(ns),1);
t_sparseLDL = zeros(length(ns),1);
t_denseLU   = zeros(length(ns),1);
t_sparseLU  = zeros(length(ns),1);
t_inv       = zeros(length(ns),1);

for ii = 1:length(ns)
    n = ns(ii);

    % Generate problem
    [H, g, A, b] = Week5_ProblemGenerator(n);
    
    % Convert to sparse for sparse solvers
    H_sp = sparse(H);
    A_sp = sparse(A);

    % Dense LDL 
    tic;
    [x_d, ~] = EqualityQPSolverLDLdense(H, g, A, b);
    t_denseLDL(ii) = toc;

    % Sparse LDL
    tic;
    [x_s, ~] = EqualityQPSolverLDLsparse(H_sp, g, A_sp, b);
    t_sparseLDL(ii) = toc;

    % Plain inverse 
    if comp_plain_inv
        tic;
        [x_inv, ~] = EqualityQPSolverPlainInverse(H, g, A, b);
        t_inv(ii) = toc;
    end

    % LU solvers 
    if comp_LU
        tic;
        [x_d, ~] = EqualityQPSolverLUdense(H, g, A, b);
        t_denseLU(ii) = toc;

        tic;
        [x_s, ~] = EqualityQPSolverLUsparse(H_sp, g, A_sp, b);
        t_sparseLU(ii) = toc;
    end

    fprintf("n = %d | Dense LDL: %.4f | Sparse LDL: %.4f | D(H): %.4f D(A): %.4f \n", ...
        n, t_denseLDL(ii), t_sparseLDL(ii), nnz(H)/n^2,nnz(A)/n^2);
end
%%
figure;

subplot(1,2,1);
plot(ns, t_denseLDL, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Problem Size n'); ylabel('Time (s)');
title('Dense LDL Time vs. n');

subplot(1,2,2);
plot(ns, t_sparseLDL, '-^', 'LineWidth', 1.5);
grid on;
xlabel('Problem Size n'); ylabel('Time (s)');
title('Sparse LDL Time vs. n');

% Match y-limits
y_max = max([t_denseLDL(:);t_sparseLDL(:)]);
subplot(1,2,1); ylim([0 y_max]);
y_max = max([t_sparseLDL(:)]);
subplot(1,2,2); ylim([0 y_max]);


figure;

subplot(1,2,1);
plot(ns, t_denseLU, '-o', 'LineWidth', 1.5);
grid on;
xlabel('Problem Size n'); ylabel('Time (s)');
title('Dense LU Time vs. n');

subplot(1,2,2);
plot(ns, t_sparseLU, '-^', 'LineWidth', 1.5);
grid on;
xlabel('Problem Size n'); ylabel('Time (s)');
title('Sparse LU Time vs. n');

y_max = max([t_denseLU(:); t_sparseLU(:)]);
subplot(1,2,1); ylim([0 y_max]);
y_max = max([t_sparseLU(:)]);
subplot(1,2,2); ylim([0 y_max]);

figure;

plot(ns, t_inv, '-o', 'LineWidth', 1.5);
grid on;

xlabel('Problem Size n'); ylabel('Time (s)');
title('Plain Inv Time vs. n');