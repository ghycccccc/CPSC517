function x = block_amg_vcycle(hierarchy, lev, b, x, nu1, nu2, smoother)
% BLOCK_AMG_VCYCLE  One block V-cycle at level lev.
%
%   Smoother: vectorized Gauss-Seidel using precomputed sweep matrices stored
%   in each hierarchy level by block_amg_setup.  Two smoother variants are
%   available, selected by the optional 7th argument:
%
%     'block'  (default) — block GS: M_fwd/M_bwd are 2x2 block lower/upper
%              triangular (diagonal 2x2 blocks + strict block-lower/upper).
%              Captures u-v coupling within each node per sweep.
%
%     'scalar' — scalar GS: M_fwd/M_bwd are the standard scalar tril/triu.
%              Treats all DOFs independently; ignores within-node coupling.
%              Uses the same block AMG hierarchy (identical P, R, A) but with
%              scalar sweep matrices stored as M_fwd_scalar_fac / M_bwd_scalar_fac.
%
%   Each sweep costs one sparse mat-vec plus one cached LU triangular solve.
%   The 2x2 block structure is preserved at every level by construction
%   (block prolongation -> coarse matrix inherits 2x2 block form).
%
% USAGE:
%   x = block_amg_vcycle(hierarchy, lev, b, x, nu1, nu2)          % block GS
%   x = block_amg_vcycle(hierarchy, lev, b, x, nu1, nu2, 'scalar') % scalar GS
%   x = block_amg_vcycle(hierarchy, lev, b, x, nu1, nu2, 'block')  % block GS

    if nargin < 7 || isempty(smoother), smoother = 'block'; end

    A = hierarchy{lev}.A;

    % Coarsest level: direct solve (no smoothing)
    if lev == length(hierarchy) || isempty(hierarchy{lev}.P)
        x = A \ b;
        return;
    end

    P = hierarchy{lev}.P;
    R = hierarchy{lev}.R;

    % Select sweep matrices based on smoother choice
    if strcmp(smoother, 'block')
        Mf   = hierarchy{lev}.M_fwd_fac;
        Mb   = hierarchy{lev}.M_bwd_fac;
        AmMf = hierarchy{lev}.AmMfwd;
        AmMb = hierarchy{lev}.AmMbwd;
    else  % 'scalar'
        Mf   = hierarchy{lev}.M_fwd_scalar_fac;
        Mb   = hierarchy{lev}.M_bwd_scalar_fac;
        AmMf = hierarchy{lev}.AmMfwd_scalar;
        AmMb = hierarchy{lev}.AmMbwd_scalar;
    end

    % Pre-smoothing: nu1 forward sweeps
    for k = 1:nu1
        x = Mf \ (b - AmMf * x);
    end

    % Restrict residual to coarse grid
    r   = b - A * x;
    r_c = R * r;

    % Recursive coarse-grid correction (start from zero for linearity)
    e_c = block_amg_vcycle(hierarchy, lev+1, r_c, zeros(size(r_c)), nu1, nu2, smoother);

    % Prolongate and correct
    x = x + P * e_c;

    % Post-smoothing: nu2 backward sweeps
    for k = 1:nu2
        x = Mb \ (b - AmMb * x);
    end
end
