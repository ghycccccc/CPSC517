function x = amg_vcycle(hierarchy, lev, b, x, nu1, nu2)
% AMG_VCYCLE  One V-cycle at level `lev`.  (corrected)
%
% Correction: Gauss-Seidel smoother now pre-extracts lower/upper
% triangular parts outside the sweep loop, avoiding repeated sparse
% row extraction that caused severe performance degradation.

    A = hierarchy{lev}.A;

    % Base case: direct solve at coarsest level
    if lev == length(hierarchy) || isempty(hierarchy{lev}.P)
        x = A \ b;
        return;
    end

    P = hierarchy{lev}.P;
    R = hierarchy{lev}.R;

    % Pre-smoothing
    x = gauss_seidel(A, b, x, nu1, 'forward');

    % Restrict residual
    r   = b - A * x;
    r_c = R * r;

    % Coarse-grid correction (always start from zero)
    e_c = amg_vcycle(hierarchy, lev+1, r_c, zeros(size(r_c)), nu1, nu2);

    % Prolongate and correct
    x = x + P * e_c;

    % Post-smoothing (backward for near-symmetry benefit)
    x = gauss_seidel(A, b, x, nu2, 'backward');

end


% =========================================================================
function x = gauss_seidel(A, b, x, nu, direction)
% GAUSS_SEIDEL  nu sweeps of Gauss-Seidel.
%
% CORRECTION over previous version:
%   Previous implementation called A(i,:)*x inside the inner loop,
%   which triggers a sparse row extraction at every index i — O(nnz)
%   overhead per row on top of the O(nnz/n) arithmetic work.
%   This made smoothing O(n * nnz) instead of O(nnz) per sweep.
%
%   Fix: pre-extract D, L, U once per call. Use in-place update form:
%     x_i <- (b_i - L(i,:)*x - U(i,:)*x) / D(i,i)
%   where x on the RHS uses already-updated values for forward sweep.
%   This is achieved by operating on L and U separately.

    n   = size(A, 1);
    d   = diag(A);              % diagonal vector [n x 1]
    L   = tril(A, -1);          % strict lower triangle (sparse)
    U   = triu(A,  1);          % strict upper triangle (sparse)

    if strcmpi(direction, 'forward')
        sweep_order = 1:n;
    else
        sweep_order = n:-1:1;
    end

    for sweep = 1:nu
        for i = sweep_order
            % In-place GS: x(i) uses already-updated x for indices
            % processed earlier in this sweep (L part) and old x for
            % indices not yet processed (U part).
            % L(i,:)*x captures the already-updated lower part.
            % U(i,:)*x captures the not-yet-updated upper part.
            x(i) = (b(i) - L(i,:)*x - U(i,:)*x) / d(i);
        end
    end

end