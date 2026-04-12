function [g, A, bl, bu, xl, xu, x_star, lambda, mu, rho, tau] = RandomLP(n, density, beta, opt)
    arguments
        n           % Number of variables
        density     % Density of sparse A matrix (0 to 1)
        beta        % Ratio of general constraints to variables
        opt.seed = []
        opt.verbose = false
    end

    % Initialize Random Number generator with seed for reproducibility
    if ~isempty(opt.seed)
        rng(opt.seed);
    end
    
    n = floor(n);
    m = floor(round(beta * n));

    % 1. Create A matrix (n x m)
    if m > 0
        A = full(sprandn(n, m, density));
    else
        A = zeros(n, 0);
    end

    % 2. Generate known optimal primal solution x_star
    x_star = randn(n, 1);

    % 3. Linear inequality constraints (bl <= A'x <= bu)
    % Randomly assign constraints to be active-low, active-high, or inactive
    Ax = A' * x_star;
    lambda = zeros(m, 1);
    mu = zeros(m, 1);
    bl = zeros(m, 1);
    bu = zeros(m, 1);

    for i = 1:m
        r = rand();
        if r < 0.33 % Active at lower bound
            bl(i) = Ax(i);
            bu(i) = Ax(i) + rand() + 0.1; 
            lambda(i) = abs(randn()); % Dual multiplier > 0
        elseif r < 0.66 % Active at upper bound
            bl(i) = Ax(i) - rand() - 0.1;
            bu(i) = Ax(i);
            mu(i) = abs(randn()); % Dual multiplier > 0
        else % Inactive
            bl(i) = Ax(i) - rand() - 0.1;
            bu(i) = Ax(i) + rand() + 0.1;
            % lambda and mu remain 0
        end
    end

    % 4. Bound constraints (xl <= x <= xu)
    rho = zeros(n, 1);
    tau = zeros(n, 1);
    xl = zeros(n, 1);
    xu = zeros(n, 1);

    for j = 1:n
        r = rand();
        if r < 0.33 % Active at lower bound
            xl(j) = x_star(j);
            xu(j) = x_star(j) + rand() + 0.1;
            rho(j) = abs(randn());
        elseif r < 0.66 % Active at upper bound
            xl(j) = x_star(j) - rand() - 0.1;
            xu(j) = x_star(j);
            tau(j) = abs(randn());
        else % Inactive
            xl(j) = x_star(j) - rand() - 0.1;
            xu(j) = x_star(j) + rand() + 0.1;
        end
    end

    % 5. Stationarity: g - A*lambda + A*mu - rho + tau = 0
    g = A * lambda - A * mu + rho - tau;

    if opt.verbose
        fprintf("Problem generated. g derived from KKT stationarity.\n");
    end
end