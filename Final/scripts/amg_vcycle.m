function x = amg_vcycle(hierarchy, lev, b, x, nu1, nu2)
% AMG_VCYCLE  One V-cycle at level lev.
%
%   Smoother: vectorized scalar Gauss-Seidel using precomputed sweep matrices
%   stored in each hierarchy level by amg_setup.  Each sweep costs one sparse
%   mat-vec plus one cached LU triangular solve — no per-row MATLAB loops.
%
%     Forward sweep:   x = M_fwd_fac \ (b - AmMfwd * x)
%     Backward sweep:  x = M_bwd_fac \ (b - AmMbwd * x)
%
%   For scalar GS (dof_per_node=1):
%     M_fwd = tril(A),   AmMfwd = triu(A, 1)   (strict upper)
%     M_bwd = triu(A),   AmMbwd = tril(A, -1)  (strict lower)
%
%   The decomposition objects are built once in amg_setup via
%   build_sweep_matrices(A, 1) and reused across all GMRES iterations
%   at zero refactoring cost.

    A = hierarchy{lev}.A;

    % Base case: direct solve at coarsest level
    if lev == length(hierarchy) || isempty(hierarchy{lev}.P)
        x = A \ b;
        return;
    end

    P = hierarchy{lev}.P;
    R = hierarchy{lev}.R;

    % Pre-smoothing: nu1 forward GS sweeps (vectorized)
    for k = 1:nu1
        x = hierarchy{lev}.M_fwd_fac \ (b - hierarchy{lev}.AmMfwd * x);
    end

    % Restrict residual to coarse grid
    r   = b - A * x;
    r_c = R * r;

    % Coarse-grid correction (always start from zero for linearity)
    e_c = amg_vcycle(hierarchy, lev+1, r_c, zeros(size(r_c)), nu1, nu2);

    % Prolongate and correct
    x = x + P * e_c;

    % Post-smoothing: nu2 backward GS sweeps (vectorized)
    for k = 1:nu2
        x = hierarchy{lev}.M_bwd_fac \ (b - hierarchy{lev}.AmMbwd * x);
    end

end
