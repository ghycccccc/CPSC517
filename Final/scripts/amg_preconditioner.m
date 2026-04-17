% =========================================================================
function z = amg_preconditioner(hierarchy, r, nu1, nu2)
% AMG_PRECONDITIONER  One V-cycle from zero — the preconditioner action.
%
% Always starts from zero. This is required for linearity:
%   M^{-1}(alpha*r) = alpha * M^{-1}(r)
% which would fail if a non-zero starting guess from a previous call
% were reused, since that guess depends on the previous (different) r.

    z = amg_vcycle(hierarchy, 1, r, zeros(size(r)), nu1, nu2);

end