% =========================================================================
%  FILE: amg_preconditioner.m
%  The preconditioner action: one V-cycle from zero.
%  This is what gets passed to GMRES.
% =========================================================================
function z = amg_preconditioner(hierarchy, r, nu1, nu2)
% AMG_PRECONDITIONER  Apply the AMG preconditioner to vector r.
%
%   z = amg_preconditioner(hierarchy, r, nu1, nu2)
%
% Computes z = M_AMG^{-1} * r by applying exactly ONE V-cycle
% starting from the zero initial guess.
%
% KEY PROPERTIES of this function:
%   1. LINEAR:  z = M^{-1} r is a linear function of r.
%               This is required for standard (non-flexible) GMRES.
%
%   2. FIXED:   The same hierarchy is used every call.
%               M^{-1} does not change across GMRES iterations.
%               This is required for the Krylov subspace to be well-defined.
%
%   3. STARTING GUESS IS ALWAYS ZERO:
%               Unlike the standalone solver (which carries x between cycles),
%               here we always start from zero because r changes every
%               GMRES iteration and there is no meaningful carry-over.
%               Starting from a non-zero guess based on a previous r would
%               break linearity and corrupt the Krylov subspace.
%
% INPUTS:
%   hierarchy  - built by amg_setup, passed in via closure (function handle)
%   r          - residual vector from GMRES at the current iteration
%   nu1, nu2   - pre- and post-smoothing sweeps (fixed throughout)
%
% OUTPUT:
%   z  - preconditioned vector; approximate solution to A*z = r

    % Always start from zero -- see theoretical note above
    x0 = zeros(size(r));

    % Apply exactly one V-cycle at the finest level
    z = amg_vcycle(hierarchy, 1, r, x0, nu1, nu2);

end