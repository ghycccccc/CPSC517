function [M_fwd_fac, M_bwd_fac, AmMfwd, AmMbwd] = build_sweep_matrices(A, dof_per_node)
% BUILD_SWEEP_MATRICES  Precompute factored forward/backward block GS sweep matrices.
%
%   For a matrix A with interleaved DOF ordering (dof_per_node DOFs per node),
%   the block Gauss-Seidel forward and backward sweeps can be written as:
%
%     Forward sweep:   x_new = M_fwd \ (b - AmMfwd * x_old)
%     Backward sweep:  x_new = M_bwd \ (b - AmMbwd * x_old)
%
%   where:
%     M_fwd  = D_block + strict_L_block   (block lower triangular part of A)
%     M_bwd  = D_block + strict_U_block   (block upper triangular part of A)
%     AmMfwd = A - M_fwd  = strict_U_block
%     AmMbwd = A - M_bwd  = strict_L_block
%
%   "Block lower/upper" is defined at the NODE level, not the scalar DOF level.
%   Node i owns DOFs  (i-1)*dof_per_node+1 : i*dof_per_node.
%   Entry A(p,q) belongs to M_fwd iff  ceil(p/d) >= ceil(q/d)  (row-node >= col-node).
%   This differs from tril(A) when dof_per_node > 1: within-diagonal-block entries
%   that sit in the scalar upper triangle (e.g. A(2i-1, 2i)) are included in M_fwd
%   because they belong to the same node (ceil(2i-1/2) == ceil(2i/2) = i).
%
%   The LU factorizations are cached via MATLAB's decomposition() object, so
%   repeated solves cost only the triangular-solve time (no refactoring).
%   All MATLAB loop overhead per DOF is eliminated: each sweep is a single
%   sparse mat-vec (AmMfwd * x) followed by a triangular solve.
%
% INPUTS:
%   A            - sparse [n x n] matrix  (n = N * dof_per_node)
%   dof_per_node - DOFs per physical node  (default 1)
%
% OUTPUTS:
%   M_fwd_fac  - decomposition object for M_fwd (cached LU factorization)
%   M_bwd_fac  - decomposition object for M_bwd (cached LU factorization)
%   AmMfwd     - sparse A - M_fwd  (used as  b - AmMfwd * x  in the RHS)
%   AmMbwd     - sparse A - M_bwd  (used as  b - AmMbwd * x  in the RHS)

    if nargin < 2 || isempty(dof_per_node), dof_per_node = 1; end

    n = size(A, 1);
    [AI, AJ, AV] = find(A);

    if dof_per_node == 1
        % Scalar case: block lower/upper coincides with standard triangular parts.
        % tril/triu are already optimal MATLAB builtins.
        M_fwd = tril(A);
        M_bwd = triu(A);
    else
        d = dof_per_node;
        row_node = ceil(AI / d);
        col_node = ceil(AJ / d);

        % M_fwd: entries where row-node >= col-node  (diagonal block + strict lower)
        fwd_mask = (row_node >= col_node);
        M_fwd = sparse(AI(fwd_mask), AJ(fwd_mask), AV(fwd_mask), n, n);

        % M_bwd: entries where row-node <= col-node  (diagonal block + strict upper)
        bwd_mask = (row_node <= col_node);
        M_bwd = sparse(AI(bwd_mask), AJ(bwd_mask), AV(bwd_mask), n, n);
    end

    % Cache LU factorizations once (UMFPACK under the hood).
    % Subsequent  M_fwd_fac \ v  calls are O(nnz) triangular solves with no
    % refactoring cost.
    M_fwd_fac = decomposition(M_fwd, 'lu');
    M_bwd_fac = decomposition(M_bwd, 'lu');

    % Precompute A - M for efficient RHS formation:  b - (A-M)*x  = b - AmM*x
    AmMfwd = A - M_fwd;   % = strict_U_block  (entries where row-node < col-node)
    AmMbwd = A - M_bwd;   % = strict_L_block  (entries where row-node > col-node)
end
