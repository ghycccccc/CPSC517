function z = geomg_preconditioner(hierarchy, r, nu1, nu2)
% GEOMG_PRECONDITIONER
% Apply one geometric multigrid V-cycle from zero initial guess.
%
% Smoothing uses vectorized block Gauss-Seidel via precomputed sweep matrices
% (M_fwd_fac, M_bwd_fac, AmMfwd, AmMbwd) stored in each hierarchy level by
% geomg_setup.  Each sweep costs one sparse mat-vec + one cached triangular
% solve — no MATLAB interpreter loops over individual DOFs or nodes.
%
%   Forward sweep:   x = M_fwd_fac \ (b - AmMfwd * x)
%   Backward sweep:  x = M_bwd_fac \ (b - AmMbwd * x)

    z = geomg_vcycle(hierarchy, 1, r, zeros(size(r)), nu1, nu2);
end


function x = geomg_vcycle(hierarchy, lev, b, x, nu1, nu2)
    A = hierarchy{lev}.A;

    % Coarsest level: direct solve (no smoothing needed)
    if lev == length(hierarchy) || isempty(hierarchy{lev}.P)
        x = A \ b;
        return;
    end

    P = hierarchy{lev}.P;
    R = hierarchy{lev}.R;

    % Pre-smoothing: nu1 forward block GS sweeps
    for k = 1:nu1
        x = hierarchy{lev}.M_fwd_fac \ (b - hierarchy{lev}.AmMfwd * x);
    end

    % Restrict residual to coarse grid
    r   = b - A * x;
    r_c = R * r;

    % Recursive coarse-grid correction (start from zero to preserve linearity)
    e_c = geomg_vcycle(hierarchy, lev + 1, r_c, zeros(size(r_c)), nu1, nu2);

    % Prolongate and add correction
    x = x + P * e_c;

    % Post-smoothing: nu2 backward block GS sweeps
    for k = 1:nu2
        x = hierarchy{lev}.M_bwd_fac \ (b - hierarchy{lev}.AmMbwd * x);
    end
end
