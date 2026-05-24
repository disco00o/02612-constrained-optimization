function [x, lambda] = EqualityQPSolver(H, g, A, b, solver)
    switch solver
        case 'LDLdense'
            [x, lambda] = EqualityQPSolverLDLdense(H,g,A,b);
        case 'LDLsparse'
            [x, lambda] = EqualityQPSolverLDLsparse(H,g,A,b);
        case 'LUdense'
            [x, lambda] = EqualityQPSolverLUdense(H,g,A,b);
        case 'LUsparse'
            [x, lambda] = EqualityQPSolverLUsparse(H,g,A,b);
        case 'plain'
            [x, lambda] = EqualityQPSolverPlainInverse(H,g,A,b);
    end
end