function x = block_amg_vcycle(hierarchy, lev, b, x, nu1, nu2)
% BLOCK_AMG_VCYCLE  One block V-cycle at level lev.
%
%   Uses block Gauss-Seidel (BGS) as the smoother: each physical node's
%   two DOFs (u_i, v_i) are updated together by solving a 2x2 system.
%   This captures the u-v coupling within each node that scalar GS misses.
%
%   Correction form of BGS:  x_i <- x_i + K_ii^{-1} * (b_i - K_i * x)
%   where K_ii is the 2x2 diagonal block and K_i * x is the full row
%   product (so the diagonal block contribution is implicitly cancelled).
%   Because x is updated in-place, preceding nodes already hold their
%   updated values when the current node is processed (standard GS property).
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

    % Pre-smoothing (forward sweep)
    x = block_gauss_seidel(A, b, x, nu1, 'forward');

    % Restrict residual to coarse grid
    r   = b - A * x;
    r_c = R * r;

    % Recursive coarse-grid correction (start from zero for linearity)
    e_c = block_amg_vcycle(hierarchy, lev+1, r_c, zeros(size(r_c)), nu1, nu2);

    % Prolongate and correct
    x = x + P * e_c;

    % Post-smoothing (backward sweep for near-symmetry benefit)
    x = block_gauss_seidel(A, b, x, nu2, 'backward');

end


% =========================================================================
function x = block_gauss_seidel(A, b, x, nu, direction)
% Block Gauss-Seidel smoother: sweeps over physical nodes (pairs of DOFs).
%
% Update formula (correction form):
%   x(rows_i) += K_ii^{-1} * (b(rows_i) - A(rows_i,:) * x)
%
% This is equivalent to the standard BGS formula
%   x_i_new = K_ii^{-1} * (b_i - sum_{j != i} K_ij * x_j)
% because K_ii^{-1} * K_ii = I cancels the diagonal-block contribution.
% In-place execution (forward or backward over nodes) provides the correct
% Gauss-Seidel property: nodes processed earlier contribute their updated
% values to subsequent nodes via A(rows_i,:)*x.
%
% Performance note: diagonal block inverses and row-pair slices of A are
% pre-extracted outside the sweep loop, avoiding repeated sparse indexing
% inside the hot loop (same pattern as the scalar GS fix in amg_vcycle.m).

    n_dofs = size(A, 1);
    N      = n_dofs / 2;

    % Pre-compute 2x2 diagonal block inverses
    D_inv = cell(N, 1);
    for i = 1:N
        ri      = 2*i-1 : 2*i;
        D_inv{i} = inv(full(A(ri, ri)));
    end

    % Pre-extract 2-row slices to avoid repeated sparse row extraction in loop
    A_rows = cell(N, 1);
    for i = 1:N
        A_rows{i} = A(2*i-1 : 2*i, :);
    end

    if strcmpi(direction, 'forward')
        node_order = 1 : N;
    else
        node_order = N : -1 : 1;
    end

    for sweep = 1 : nu
        for i = node_order
            ri    = 2*i-1 : 2*i;
            r_i   = b(ri) - A_rows{i} * x;       % residual at node i (in-place x)
            x(ri) = x(ri) + D_inv{i} * r_i;      % block correction
        end
    end

end
