% =========================================================================
%  FILE: amg_vcycle.m
%  The V-cycle: the shared computational kernel used by BOTH the
%  standalone solver and the preconditioner.
% =========================================================================
function x = amg_vcycle(hierarchy, lev, b, x, nu1, nu2)
% AMG_VCYCLE  Apply one V-cycle at level `lev`.
%
%   x = amg_vcycle(hierarchy, lev, b, x, nu1, nu2)
%
% INPUTS:
%   hierarchy  - struct array built by amg_setup
%   lev        - current level index (1 = finest)
%   b          - right-hand side at this level
%   x          - initial guess at this level
%                  * Standalone solver: carry-over from previous cycle
%                  * Preconditioner:    always zeros(size(b))
%   nu1, nu2   - number of pre- and post-smoothing sweeps
%
% OUTPUT:
%   x  - improved approximation to A^{-1} b at this level
%
% THEORETICAL ROLE:
%   One application computes z = M_AMG^{-1} r, the preconditioner action.
%   The map r -> z is LINEAR and FIXED (does not depend on iteration count),
%   which is the requirement for use as a standard GMRES preconditioner.

    A = hierarchy{lev}.A;

    % ---- Base case: coarsest level -> direct solve ----
    if lev == length(hierarchy) || isempty(hierarchy{lev}.P)
        x = A \ b;
        return;
    end

    P = hierarchy{lev}.P;
    R = hierarchy{lev}.R;

    % ------------------------------------------------------------------
    % PRE-SMOOTHING: nu1 sweeps of Gauss-Seidel
    % Solves A*x = b approximately, starting from current x.
    %
    % One GS sweep: for each i,  x_i <- (b_i - sum_{j~=i} a_{ij} x_j) / a_{ii}
    % This is a FORWARD sweep (indices 1..n in order).
    % ------------------------------------------------------------------
    x = gauss_seidel(A, b, x, nu1, 'forward');

    % ------------------------------------------------------------------
    % RESTRICT RESIDUAL to coarse grid
    % r = b - A*x  (fine-grid residual)
    % r_c = R * r  (coarse-grid residual)
    % ------------------------------------------------------------------
    r   = b - A * x;
    r_c = R * r;

    % ------------------------------------------------------------------
    % COARSE-GRID CORRECTION (recursive V-cycle)
    % Solve A_c * e_c = r_c approximately.
    % Starting guess for coarse correction is always zero.
    % ------------------------------------------------------------------
    e_c = amg_vcycle(hierarchy, lev + 1, r_c, zeros(size(r_c)), nu1, nu2);

    % ------------------------------------------------------------------
    % PROLONGATE correction and UPDATE fine-grid approximation
    % x <- x + P * e_c
    % ------------------------------------------------------------------
    x = x + P * e_c;

    % ------------------------------------------------------------------
    % POST-SMOOTHING: nu2 sweeps of Gauss-Seidel (backward sweep)
    % Using a backward sweep for post-smoothing is standard practice:
    % it makes the combined smoother symmetric when A is symmetric,
    % which helps convergence even for near-symmetric problems.
    % ------------------------------------------------------------------
    x = gauss_seidel(A, b, x, nu2, 'backward');

end

% =========================================================================
%  LOCAL FUNCTION: Gauss-Seidel smoother
% =========================================================================
function x = gauss_seidel(A, b, x, nu, direction)
% GAUSS_SEIDEL  Apply nu sweeps of Gauss-Seidel smoothing.
%
% For each sweep, iterates over all indices in either forward or backward
% order and updates:
%   x_i <- (b_i - sum_{j~=i} a_{ij} x_j) / a_{ii}
%
% This is implemented efficiently using the sparse structure of A.
% We split A = D + L + U and use the in-place update form.
%
% 'forward'  sweep: i = 1, 2, ..., n
% 'backward' sweep: i = n, n-1, ..., 1

    n   = size(A, 1);
    d   = diag(A);          % diagonal entries (n x 1 vector)

    if strcmpi(direction, 'forward')
        sweep_order = 1:n;
    else
        sweep_order = n:-1:1;
    end

    for sweep = 1:nu
        for i = sweep_order
            % Extract row i of A (excluding diagonal)
            % A(i,:)*x gives the full dot product; subtract diagonal contribution
            residual_i = b(i) - A(i,:) * x + d(i) * x(i);
            x(i)       = residual_i / d(i);
        end
    end

end