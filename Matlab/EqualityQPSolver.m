function [x, lambda] = EqualityQPSolver(H, g, A, b, solver)
    switch solver
        case 'LDLdense'
            [x, lambda] = EqualityQPSolverLDLdense(H,g,A,b);
        case 'LDLsparse'
            [x, lambda] = EqualityQPSolverLDLsparse(H,g,A,b);
    end
end