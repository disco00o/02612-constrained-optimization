function [H, g, A, b, H_d, A_d] = RandomEQP(n, alpha, density, beta, flag, opt)
    arguments
        n 
        alpha 
        density 
        beta 
        flag = 'dense'
        opt.seed = []
        opt.verbose = false
    end


    % Initialize Random Number generator with seed for reproducability
    if ~isempty(opt.seed)
        rng(opt.seed);
    end

    n = floor(n);
    % Get Columns 
    m = floor(round(beta * n));
    % if m > 0
    %     A = sprandn(n, m, density);
    %     tic
    % 
    %     % Fast sparse check using QR decomposition
    %     R = qr(A, 0); 
    %     while any(abs(diag(R)) < 1e-12)
    %         A = sprandn(n, m, density);
    %         R = qr(A, 0);
    %     end
    % 
    %     fprintf("Took %f seconds to find a suitable A matrix that has full rank! \n", toc)
    %     if opt.verbose, fprintf("A has %f non-zero elements! \n", nnz(A)/(n*m)); end
    %     b = -1 + 2 * rand(m, 1);

    if m > 0
        % Create A matrix (Note, the constrained reads A'*x = b)
        % -> Drawing from a normal distribution
        % -> It will be generated in sparse format
        % Inside RandomEQP, instead of using the raw density for A:
        %fixed_nnz_per_col = 5; 
        %A_density = fixed_nnz_per_col / n; 
        A = sprandn(n, m, density);

        %A = sprandn(n, m, density);

        tic
        while rank(full(A)) < m
            A = sprandn(n, m, density);
        end
        fprintf("\n A has a density of: %f \n", nnz(A)/n^2)
        fprintf("Took %f seconds to find a suitable A matrix that has full rank! \n", toc)
        if opt.verbose, fprintf("A has %f non-zero elements! \n",nnz(A)/(n*m)); end
        b = -1 + 2 * rand(m, 1);
    else
        A = sparse(n, 0); % Empty Sparse-Matrix
        b = zeros(0, 1);  % Empty vector
    end

    M = sprandn(n, n, density);
    if opt.verbose, fprintf("M has %f non-zero elements! \n",nnz(M)/(n^2)); end
    H = M * M' + alpha *  speye(n); % <- This makes H increasingly dense for higher n!
    %H = sprandsym(n, 0.005) + alpha *  speye(n);
    fprintf("\n H has a density of: %f \n", nnz(H)/n^2)
    
    % Create g and b
    % -> Drawing from a uniform distribution in the interval -1 to 1
    g = -1 + 2 * rand(n, 1);

    % Convert to dense 'normal' matrix, if desired
    switch flag
        case 'dense'
            H = full(H);
            A = full(A);
            assert(all(eig(H) > 0));
        case 'sparse'
            H_d = full(H);
            %assert(all(eig(H_d) > 0), 'Non convex problem generated!');
        case 'sparse&dense'
            H_d = full(H);
            %assert(all(eig(H_d) > 0), 'Non convex problem generated!');
            A_d = full(A);
            return;
        otherwise
            fprintf('Invalid Flag!\n');
    end
    
    if ~exist('H_d', 'var'), H_d = []; end
    if ~exist('A_d', 'var'), A_d = []; end
    
end
