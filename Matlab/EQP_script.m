clear all; clc; close all;
% 1: Equality Constrained Convex QP
% 1.3) Random EQP Generator for dense and sparse matrices
% Generate example problem

[H_d, g, A_d, b] = RandomEQP(100, 0.1, 0.15, 0.5, 'dense', seed=4444, verbose=true);
[H_sp, g, A_sp, b] = RandomEQP(100, 0.1, 0.15, 0.5, 'sparse', seed=4444, verbose=true);
[H_sp, g, A_sp, b, H_d, A_d] = RandomEQP(100, 0.1, 0.15, 0.5, 'sparse&dense', seed=4444, verbose=true);

% 1.4 Sparse & Dense Solvers
[x_d, lambda_d] = EqualityQPSolverLDLdense(H_d,g,A_d,b)
[x_d, lambda_d] = EqualityQPSolver(H_d,g,A_d,b,'LDLdense')
[x_sp, lambda_sp] = EqualityQPSolverLDLsparse(H_sp,g,A_sp,b)
[x_sp, lambda_sp] = EqualityQPSolver(H_sp,g,A_sp,b,'LDLsparse')

% 1.5 Time comparison
ns = 10:100:800;
betas = 0:0.5:1;

seed = 4444;
alpha = 1000;
density = 0.15;

i = length(ns)
j = length(betas)
t_dense = zeros(i, j);
t_sparse = zeros(i, j);


for i = 1:length(ns)

    n = ns(i)
    for j = 1:length(betas)
        beta = betas(j);
        
        [H_sp, g, A_sp, b, H_d, A_d] = RandomEQP(n, alpha, density, beta, 'sparse&dense', seed=seed);
        
        tic;
        [x_d, ~] = EqualityQPSolverLDLdense(H_d, g, A_d, b);
        t_dense(i,j) = toc;
        
        tic;
        [x_sp, ~] = EqualityQPSolverLDLsparse(H_sp, g, A_sp, b);
        t_sparse(i,j) = toc;

        fprintf("\n n: %f beta:%f ; \n DenseT: %f \n SparseT: %f \n",n,beta,t_dense(i,j),t_sparse(i,j))
    end
end


%% Plotting the Surfaces
figure('Name', 'Solver Time Comparison', 'Color', 'w');

% Create a grid for the plot
[BETA, N] = meshgrid(betas, ns);

% Subplot 1: Dense Solver
subplot(1,2,1);
surf(N, BETA, t_dense);
xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
title('LDL Dense Solver Time');
shading interp; colormap jet; colorbar;

% Subplot 2: Sparse Solver
subplot(1,2,2);
surf(N, BETA, t_sparse);
xlabel('Problem Size n'); ylabel('Constraint Ratio \beta'); zlabel('Time (s)');
title('LDL Sparse Solver Time');
shading interp; colormap jet; colorbar;

% Match the Z-axis scales for fair comparison
z_max = max([max(t_dense(:)), max(t_sparse(:))]);
subplot(1,2,1); zlim([0 z_max]);
subplot(1,2,2); zlim([0 z_max]);


