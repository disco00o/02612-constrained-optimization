clear all; clc; close all;
% 1: Equality Constrained Convex QP
% 1.3) Random EQP Generator for dense and sparse matrices
% Generate example problem

[H_d, g, A_d, b] = RandomEQP(10, 0.1, 0.15, 0.5, 'dense', seed=4444, verbose=true);
[H_sp, g, A_sp, b] = RandomEQP(10, 0.1, 0.15, 0.5, 'sparse', seed=4444, verbose=true);
[H_sp, g, A_sp, b, H_d, A_d] = RandomEQP(100, 0.1, 0.15, 0.5, 'sparse&dense', seed=4444, verbose=true);

% 1.4 Sparse & Dense Solvers
[x_d, lambda_d] = EqualityQPSolverLDLdense(H_d,g,A_d,b)
%[x_d, lambda_d] = EqualityQPSolver(H_d,g,A_d,b,'LDLdense')
[x_sp, lambda_sp] = EqualityQPSolverLDLsparse(H_sp,g,A_sp,b)
%[x_sp, lambda_sp] = EqualityQPSolver(H_sp,g,A_sp,b,'LDLsparse')

% Plain solver
[x_inv_LDL, lambda_inv] = EqualityQPSolverLDLdense(H_d,g,A_d,b)
[x_inv_LU, lambda_inv] = EqualityQPSolverLUdense(H_d,g,A_d,b)

%%
clc;
% 1.5 Time comparison

comp_plain_inv = true;
comp_LU = true;

%ns = [10:100:800 800];
ns = 100:100:1000; % 10:10:800;
betas = 0:0.25:1;

seed = 4444;
alpha = 100; % -00
density = 0.15; % 0.15

i = length(ns)
j = length(betas)
t_denseLDL = zeros(i, j);
t_sparseLDL = zeros(i, j);
t_denseLU = zeros(i, j);
t_sparseLU = zeros(i, j);
t_inv = zeros(i, j);
density_H = zeros(i,j);

for i = 1:length(ns)

    n = ns(i);
    for j = 1:length(betas)
        beta = betas(j);
        
        [H_sp, g, A_sp, b, H_d, A_d] = RandomEQP(n, alpha, density, beta, 'sparse&dense', seed=seed);
        
        density_H(i,j) = nnz(H_d)/n^2;

        tic;
        [x_d, ~] = EqualityQPSolverLDLdense(H_d, g, A_d, b);
        t_denseLDL(i,j) = toc;
                
        tic;
        [x_sp, ~] = EqualityQPSolverLDLsparse(H_sp, g, A_sp, b);
        t_sparseLDL(i,j) = toc;

        fprintf("\n n: %f beta:%f ; \n DenseT: %f \n SparseT: %f \n",n,beta,t_denseLDL(i,j),t_sparseLDL(i,j))
        
        if comp_plain_inv
            tic;
            [x_inv, ~] = EqualityQPSolverPlainInverse(H_d, g, A_d, b);
            t_inv(i,j) = toc;
        end
        if comp_LU
            tic;
            [x_d, ~] = EqualityQPSolverLUdense(H_d, g, A_d, b);
            t_denseLU(i,j) = toc;
                    
            tic;
            [x_sp, ~] = EqualityQPSolverLUsparse(H_sp, g, A_sp, b);
            t_sparseLU(i,j) = toc;
        end
    
    end
end


%% Plotting the Surfaces
close all;


% Create a grid for the plot
[BETA, N] = meshgrid(betas, ns);

% Plot density of H
figure('Name', 'EQ-Problem Comparison - Density of H', 'Color', 'w');
        surf(N, BETA, density_H);
        xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
        title('Density of H');
        shading interp; colormap jet; colorbar;
        %zlim([0 z_max]);

figure('Name', 'Solver Time Comparison - LDL', 'Color', 'w');


% Subplot 1: Dense Solver
% if ~comp_plain_inv
%     subplot(1,2,1);
%     surf(N, BETA, t_denseLDL);
%     xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
%     title('LDL Dense Solver Time');
%     shading interp; colormap jet; colorbar;
% 
%     % Subplot 2: Sparse Solver
%     subplot(1,2,2);
%     surf(N, BETA, t_sparseLDL);
%     xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
%     title('LDL Sparse Solver Time');
%     shading interp; colormap jet; colorbar;
% 
%     % Match the Z-axis scales for fair comparison
%     z_max = max([max(t_denseLDL(:)), max(t_sparseLDL(:))]);
%     subplot(1,2,1); zlim([0 z_max]);
%     subplot(1,2,2); zlim([0 z_max]);
% 
% else
    subplot(1,2,1);
    surf(N, BETA, t_denseLDL);
    xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
    title('LDL Dense Solver Time');
    shading interp; colormap jet; colorbar;
    
    % Subplot 2: Sparse Solver
    subplot(1,2,2);
    surf(N, BETA, t_sparseLDL);
    xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
    title('LDL Sparse Solver Time');
    shading interp; colormap jet; colorbar;

    % Match the Z-axis scales for fair comparison
    z_max = max([max(t_denseLDL(:)), max(t_sparseLDL(:)),max(t_inv(:))]);
    subplot(1,2,1); zlim([0 z_max]);
    subplot(1,2,2); zlim([0 z_max]);
    
    if comp_plain_inv
        figure('Name', 'Solver Time Comparison - plain inverse', 'Color', 'w');
        surf(N, BETA, t_inv);
        xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
        title('Plain Inverse Solver Time');
        shading interp; colormap jet; colorbar;
        zlim([0 z_max]);
    end

    if comp_LU
        figure('Name', 'Solver Time Comparison - LU', 'Color', 'w');
        subplot(1,2,1);
        surf(N, BETA, t_denseLU);
        xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
        title('LU Dense Solver Time');
        shading interp; colormap jet; colorbar;
        
        % Subplot 2: Sparse Solver
        subplot(1,2,2);
        surf(N, BETA, t_sparseLU);
        xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
        title('LU Sparse Solver Time');
        shading interp; colormap jet; colorbar;
    
        % Match the Z-axis scales for fair comparison
        subplot(1,2,1); zlim([0 z_max]);
        z_max = max([max(t_denseLDL(:)), max(t_sparseLDL(:)),max(t_denseLU(:)), max(t_sparseLU(:)),max(t_inv(:))]);
        
        subplot(1,2,2); zlim([0 z_max]);
    end

%end

%% Additional Plot: Computational Time vs. n for Different Betas
figure('Name', 'Solver Performance Analysis by Constraint Ratio \beta - LDL', 'Color', 'w');

% Extract the unique values from your meshgrid setup
unique_betas = unique(BETA); 
n_axis = unique(N); % The X-axis data representing problem sizes

% --- Subplot 1: Dense LDL Solver ---
subplot(1, 2, 1);
hold on;
for i = 1:length(unique_betas)
    current_beta = unique_betas(i);
    % Logical indexing to pull times corresponding to the current beta
    idx = (BETA == current_beta);
    
    plot(n_axis, t_denseLDL(idx), '-o', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('\\beta = %.2f', current_beta));
end
hold off;
grid on;
xlabel('Problem Size n'); ylabel('Time (s)');
title('Dense LDL Time vs. n');
legend('Location', 'best');

% --- Subplot 2: Sparse LDL Solver ---
subplot(1, 2, 2);
hold on;
for i = 1:length(unique_betas)
    current_beta = unique_betas(i);
    idx = (BETA == current_beta);
    
    plot(n_axis, t_sparseLDL(idx), '-^', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('\\beta = %.2f', current_beta));
end
hold off;
grid on;
xlabel('Problem Size n'); ylabel('Time (s)');
title('Sparse LDL Time vs. n');
legend('Location', 'best');

% Match the Y-axis scales for a fair comparison
y_max = max([max(t_denseLDL(:)), max(t_sparseLDL(:))]);
subplot(1, 2, 1); ylim([0 y_max]);
subplot(1, 2, 2); ylim([0 y_max]);


figure('Name', 'Solver Performance Analysis by Constraint Ratio \beta - LU', 'Color', 'w');

% Extract the unique values from your meshgrid setup
unique_betas = unique(BETA); 
n_axis = unique(N); % The X-axis data representing problem sizes

% --- Subplot 1: Dense LDL Solver ---
subplot(1, 2, 1);
hold on;
for i = 1:length(unique_betas)
    current_beta = unique_betas(i);
    % Logical indexing to pull times corresponding to the current beta
    idx = (BETA == current_beta);
    
    plot(n_axis, t_denseLU(idx), '-o', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('\\beta = %.2f', current_beta));
end
hold off;
grid on;
xlabel('Problem Size n'); ylabel('Time (s)');
title('Dense LU Time vs. n');
legend('Location', 'best');

% --- Subplot 2: Sparse LDL Solver ---
subplot(1, 2, 2);
hold on;
for i = 1:length(unique_betas)
    current_beta = unique_betas(i);
    idx = (BETA == current_beta);
    
    plot(n_axis, t_sparseLU(idx), '-^', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('\\beta = %.2f', current_beta));
end
hold off;
grid on;
xlabel('Problem Size n'); ylabel('Time (s)');
title('Sparse LU Time vs. n');
legend('Location', 'best');

% Match the Y-axis scales for a fair comparison
subplot(1, 2, 1); ylim([0 y_max]);
y_max = max([max(t_denseLU(:)), max(t_sparseLU(:))]);
subplot(1, 2, 2); ylim([0 y_max]);

