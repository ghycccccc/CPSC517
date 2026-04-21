function x = block_amg_vcycle(hierarchy, lev, b, x, nu1, nu2)
% BLOCK_AMG_VCYCLE  One block V-cycle at level lev.
%
%   Smoother: vectorized block Gauss-Seidel using precomputed sweep matrices
%   stored in each hierarchy level by block_amg_setup.  Each sweep is a
%   single sparse mat-vec plus a cached LU solve — no per-node MATLAB loops.
%
%     Forward sweep:   x = M_fwd_fac \ (b - AmMfwd * x)
%     Backward sweep:  x = M_bwd_fac \ (b - AmMbwd * x)
%
%   M_fwd is the block lower-triangular part of A (diagonal 2x2 blocks +
%   strict block-lower), M_bwd is the block upper-triangular part.  Both
%   are factored once in block_amg_setup via decomposition(...,'lu').
%
%   The 2x2 block structure is preserved at every level by construction
%   (block prolongation -> coarse matrix inherits 2x2 block form).

    A = hierarchy{lev}.A;

    % Coarsest level: direct solve
    if lev == length(hierarchy) || isempty(hierarchy{lev}.P)
        x = A \ b;
        return;
    end

    P = hierarchy{lev}.P;
    R = hierarchy{lev}.R;

    % Pre-smoothing: nu1 forward block GS sweeps (vectorized)
    for k = 1:nu1
        x = hierarchy{lev}.M_fwd_fac \ (b - hierarchy{lev}.AmMfwd * x);
    end

    % Restrict residual to coarse grid
    r   = b - A * x;
    r_c = R * r;

    % Recursive coarse-grid correction (start from zero for linearity)
    e_c = block_amg_vcycle(hierarchy, lev+1, r_c, zeros(size(r_c)), nu1, nu2);

    % Prolongate and correct
    x = x + P * e_c;

    % Post-smoothing: nu2 backward block GS sweeps (vectorized)
    for k = 1:nu2
        x = hierarchy{lev}.M_bwd_fac \ (b - hierarchy{lev}.AmMbwd * x);
    end
end
