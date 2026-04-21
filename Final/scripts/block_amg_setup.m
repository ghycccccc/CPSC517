function hierarchy = block_amg_setup(K, theta, max_levels, coarse_threshold)
% BLOCK_AMG_SETUP  Build block (2x2) AMG hierarchy for vector-field problems.
%
%   Treats the [2N x 2N] matrix K as N physical nodes, each holding 2 coupled
%   DOFs (u_i, v_i) at interleaved positions (2i-1, 2i).  All coarsening
%   decisions operate at the NODE level, so both DOFs of a node are always
%   coarsened together.  This avoids the scalar-AMG failure mode of splitting
%   u_i and v_i into opposite C/F classes, which destroys the local coupling.
%
%   Prolongation is built in two stages (SA-AMG style):
%
%   Stage 1 — Tentative P (block_prolongation):
%     - C-nodes:  identity I_2 (exact injection of both components).
%     - F-nodes:  omega_ij = s_ij / sum_{k in C_i} s_ik  (proportional to
%                 Frobenius-norm coupling strength), summing to 1 per F-node.
%                 Partition-of-unity weights preserve constant (rigid-body
%                 translation) modes exactly on the tentative grid.
%
%   Stage 2 — Prolongation smoothing (smooth_prolongation):
%     P_smooth = P_tent - omega * D_block^{-1} * A * P_tent,
%     omega = 4 / (3 * rho(D_block^{-1} * A))  (estimated by power iter).
%     This enforces A * P_smooth * e_c ≈ 0 for smooth error modes, which
%     is the AMG interpolation property violated by Stage 1 alone.
%
%   The 2x2 block structure is preserved at every coarse level because the
%   Galerkin operator R * A * P maps paired DOFs to paired coarse DOFs.
%
% INPUTS:
%   K                 - sparse [2N x 2N] system (interleaved DOF ordering)
%   theta             - strong dependence threshold (default 0.25)
%   max_levels        - max hierarchy depth (default 10)
%   coarse_threshold  - stop coarsening when node count <= this (default 40)
%
% OUTPUT:
%   hierarchy{l}.A  - matrix at level l  (2*N_l x 2*N_l)
%   hierarchy{l}.P  - block prolongation ([] at coarsest level)
%   hierarchy{l}.R  - restriction = P'   ([] at coarsest level)

    if nargin < 2 || isempty(theta),            theta            = 0.25; end
    if nargin < 3 || isempty(max_levels),       max_levels       = 10;   end
    if nargin < 4 || isempty(coarse_threshold), coarse_threshold = 40;   end

    hierarchy    = {};
    hierarchy{1} = struct('A', K, 'P', [], 'R', []);
    [hierarchy{1}.M_fwd_fac, hierarchy{1}.M_bwd_fac, ...
     hierarchy{1}.AmMfwd,    hierarchy{1}.AmMbwd] = ...
        build_sweep_matrices(K, 2);
    [hierarchy{1}.M_fwd_scalar_fac, hierarchy{1}.M_bwd_scalar_fac, ...
     hierarchy{1}.AmMfwd_scalar,    hierarchy{1}.AmMbwd_scalar] = ...
        build_sweep_matrices(K, 1);

    for lev = 1 : max_levels - 1

        A_lev  = hierarchy{lev}.A;
        n_dofs = size(A_lev, 1);
        N      = n_dofs / 2;           % physical nodes at this level

        if N <= coarse_threshold, break; end

        % Step 1: N x N node-level strength matrix
        %   S(i,j) = Frobenius norm of 2x2 block A(2i-1:2i, 2j-1:2j)
        S_nodes = node_strength_matrix(A_lev, N);

        % Step 2: strong connections and C/F splitting (on nodes)
        [S_strong, ST_strong] = node_strong_dependence(S_nodes, N, theta);
        [C_nodes, ~]          = cf_splitting(S_strong, ST_strong, N);

        if sum(C_nodes) == 0 || sum(C_nodes) == N, break; end

        % Step 3: block prolongation P (tentative)  [2N x 2*Nc]
        P = block_prolongation(S_nodes, C_nodes, S_strong, N);

        if isempty(P) || size(P, 2) == 0, break; end

        % Step 4: SA-AMG prolongation smoothing
        %   P_smooth = (I - omega * D_block^{-1} * A) * P_tent
        %
        %   The tentative P has proportional weights that preserve constant
        %   near-null modes (partition of unity) but do not satisfy the AMG
        %   interpolation property for the actual smooth error of A.  One
        %   damped block-Jacobi step projects out the fine-scale components
        %   from the prolongation columns, giving a P_smooth that better
        %   represents the slow modes of A on the coarse grid.
        P = smooth_prolongation(A_lev, P, N);

        R        = P';
        A_coarse = R * A_lev * P;

        hierarchy{lev}.P   = P;
        hierarchy{lev}.R   = R;
        hierarchy{lev+1}   = struct('A', A_coarse, 'P', [], 'R', []);
        [hierarchy{lev+1}.M_fwd_fac, hierarchy{lev+1}.M_bwd_fac, ...
         hierarchy{lev+1}.AmMfwd,    hierarchy{lev+1}.AmMbwd] = ...
            build_sweep_matrices(A_coarse, 2);
        [hierarchy{lev+1}.M_fwd_scalar_fac, hierarchy{lev+1}.M_bwd_scalar_fac, ...
         hierarchy{lev+1}.AmMfwd_scalar,    hierarchy{lev+1}.AmMbwd_scalar] = ...
            build_sweep_matrices(A_coarse, 1);

    end
end


% =========================================================================
function S = node_strength_matrix(A, N)
% N x N node-level strength matrix.
% S(i,j) = Frobenius norm of the 2x2 off-diagonal block A(2i-1:2i, 2j-1:2j).
%
% Uses vectorised sparse accumulation of squared entry values to avoid an
% explicit double loop over node pairs.

    [AI, AJ, AV] = find(A);

    % Map DOF index to node index (node i owns DOFs 2i-1 and 2i)
    ni = ceil(AI / 2);
    nj = ceil(AJ / 2);

    % Keep only entries belonging to off-diagonal blocks
    off  = (ni ~= nj);
    ni   = ni(off);
    nj   = nj(off);
    av2  = AV(off) .^ 2;

    % sparse() sums duplicate (ni,nj) pairs -> sum of squares per block
    S_sq = sparse(ni, nj, av2, N, N);

    % Element-wise sqrt of nonzeros -> Frobenius norm per block
    [si, sj, sv] = find(S_sq);
    S = sparse(si, sj, sqrt(sv), N, N);
end


% =========================================================================
function [S_strong, ST_strong] = node_strong_dependence(S_nodes, N, theta)
% Strong-connection sets for the node-level strength graph.
% i -> j is strong if S(i,j) >= theta * max_{k != i} S(i,k).
% All entries of S_nodes are non-negative (Frobenius norms), so no sign
% filtering is needed (unlike the scalar AMG case for M-matrices).

    S_strong  = cell(N, 1);
    ST_strong = cell(N, 1);
    for k = 1:N, ST_strong{k} = []; end

    for i = 1:N
        row_i    = full(S_nodes(i, :));
        row_i(i) = 0;
        max_s    = max(row_i);

        if max_s <= 0
            S_strong{i} = [];
            continue;
        end

        strong_j    = find(row_i >= theta * max_s);
        S_strong{i} = strong_j;

        for j = strong_j
            ST_strong{j} = [ST_strong{j}, i];
        end
    end
end


% =========================================================================
function [C_pts, F_pts] = cf_splitting(S, ST, n)
% Greedy C/F splitting with H-1 enforcement, applied to the node-level graph.
% Algorithm identical to amg_setup.m cf_splitting.

    status = zeros(n, 1);     % 0 = unassigned, 1 = C, -1 = F
    lambda = cellfun(@length, ST);

    % First pass: greedy coloring
    while true
        unassigned = find(status == 0);
        if isempty(unassigned), break; end

        [~, idx] = max(lambda(unassigned));
        i_star   = unassigned(idx);
        status(i_star) = 1;

        for j = ST{i_star}(:)'
            if status(j) == 0
                status(j) = -1;
                for k = S{j}(:)'
                    if status(k) == 0
                        lambda(k) = lambda(k) + 1;
                    end
                end
            end
        end
    end

    % Isolated nodes (no strong connections) become C-points
    status(status == 0) = 1;

    % Second pass: H-1 enforcement
    % For each F-node i, every strong neighbor j must be a C-node or
    % must strongly depend on some C-node that i also strongly depends on.
    for i = find(status == -1)'
        C_i = intersect(S{i}, find(status == 1));
        for j = S{i}(:)'
            if status(j) == 1, continue; end
            if isempty(intersect(S{j}, C_i))
                status(j) = 1;
                C_i = [C_i; j]; 
            end
        end
    end

    C_pts = (status ==  1);
    F_pts = (status == -1);
end


% =========================================================================
function P = smooth_prolongation(A, P_tent, N)
% SA-AMG prolongation smoother: one damped block-Jacobi step.
%
%   P_smooth = P_tent - omega * D_block^{-1} * A * P_tent
%
%   D_block is the block-diagonal of A (2x2 blocks); its inverse is built
%   as a sparse matrix so that the product D_block^{-1} * A * P_tent is a
%   single sparse-sparse-dense multiply rather than N small solves inside a
%   loop.  omega = 4 / (3 * rho), where rho is estimated by 10 power
%   iterations on D_block^{-1} * A.  omega is clamped to [0.1, 1.0] so
%   that a bad spectral-radius estimate cannot make P_smooth degenerate.
%
%   Effect on the AMG interpolation property:
%     The tentative P satisfies the partition-of-unity condition (constant
%     vectors are interpolated exactly), but proportional weights do not
%     satisfy A * e ≈ 0 for smooth error e.  The Jacobi step enforces
%                 A * P_smooth * e_c ≈ 0
%     in the sense that the residual ‖A * P_smooth * e_c‖ is minimised
%     over one Jacobi step — the same rationale as in energy-minimising AMG.

    n_dofs = size(A, 1);

    % ------------------------------------------------------------------
    % Build sparse D_block_inv: 2x2 block-diagonal inverse of A
    % ------------------------------------------------------------------
    II  = zeros(4*N, 1);
    JJ  = zeros(4*N, 1);
    VV  = zeros(4*N, 1);
    ptr = 0;
    for i = 1:N
        ri       = 2*i-1 : 2*i;
        D_blk    = full(A(ri, ri));
        det_D    = D_blk(1,1)*D_blk(2,2) - D_blk(1,2)*D_blk(2,1);
        if abs(det_D) < 1e-14
            % Degenerate diagonal block: fall back to scalar inverse
            d_avg = (abs(D_blk(1,1)) + abs(D_blk(2,2))) / 2;
            d_avg = max(d_avg, 1e-14);
            D_inv = eye(2) / d_avg;
        else
            D_inv = [D_blk(2,2), -D_blk(1,2); -D_blk(2,1), D_blk(1,1)] / det_D;
        end
        for r = 1:2
            for c = 1:2
                ptr       = ptr + 1;
                II(ptr)   = ri(r);
                JJ(ptr)   = ri(c);
                VV(ptr)   = D_inv(r, c);
            end
        end
    end
    D_block_inv = sparse(II, JJ, VV, n_dofs, n_dofs);

    % ------------------------------------------------------------------
    % Estimate spectral radius of D_block_inv * A via power iteration
    % ------------------------------------------------------------------
    D_inv_A = D_block_inv * A;   % sparse x sparse -> sparse

    v = ones(n_dofs, 1) / sqrt(n_dofs);
    rho = 1.0;
    for k = 1:10
        w   = D_inv_A * v;
        rho = norm(w);
        if rho < 1e-14, break; end
        v   = w / rho;
    end
    rho = max(rho, 0.1);   % guard against degenerate estimate

    % omega = 4/(3*rho), clamped to a safe range
    omega = min(max(4 / (3 * rho), 0.10), 1.0);

    % ------------------------------------------------------------------
    % Smoothed prolongation
    % ------------------------------------------------------------------
    P = P_tent - omega * (D_inv_A * P_tent);

end


% =========================================================================
function P = block_prolongation(S_nodes, C_nodes, S_strong, N)
% Build block prolongation P of size [2N x 2*Nc].
%
%   C-node i (coarse index ki): P(2i-1:2i, 2ki-1:2ki) = I_2.
%   F-node i interpolating from strong C-neighbors {j}:
%     P(2i-1:2i, 2kj-1:2kj) = omega_ij * I_2,
%     omega_ij = S(i,j) / sum_{k in C_i} S(i,k).
%
%   Proportional weights sum to 1 per F-node (partition of unity),
%   which exactly preserves the constant near-null mode.

    Nc = sum(C_nodes);
    coarse_idx          = zeros(N, 1);
    coarse_idx(C_nodes) = 1 : Nc;

    % Conservative pre-allocation (2 DOFs per node, up to ~9 C-neighbors)
    max_entries = 2 * N * 20;
    PI = zeros(max_entries, 1);
    PJ = zeros(max_entries, 1);
    PV = zeros(max_entries, 1);
    ptr = 0;

    for i = 1:N

        if C_nodes(i)

            ki = coarse_idx(i);
            ptr = ptr + 1;  PI(ptr) = 2*i-1;  PJ(ptr) = 2*ki-1;  PV(ptr) = 1.0;
            ptr = ptr + 1;  PI(ptr) = 2*i;    PJ(ptr) = 2*ki;    PV(ptr) = 1.0;

        else

            Ni  = S_strong{i};
            C_i = Ni(C_nodes(Ni));

            if isempty(C_i)
                % Fallback: promote this F-node to C (same logic as amg_setup.m)
                C_nodes(i)    = true;
                Nc            = Nc + 1;
                coarse_idx(i) = Nc;
                ki = Nc;
                ptr = ptr + 1;  PI(ptr) = 2*i-1;  PJ(ptr) = 2*ki-1;  PV(ptr) = 1.0;
                ptr = ptr + 1;  PI(ptr) = 2*i;    PJ(ptr) = 2*ki;    PV(ptr) = 1.0;
                continue;
            end

            % Proportional weights: omega_ij = s_ij / sum_{k in C_i} s_ik
            strengths = full(S_nodes(i, C_i));
            total_s   = sum(strengths);

            if total_s < 1e-14
                strengths = ones(1, length(C_i));
                total_s   = length(C_i);
            end

            for jj = 1 : length(C_i)
                j     = C_i(jj);
                omega = strengths(jj) / total_s;
                ki    = coarse_idx(j);
                ptr = ptr + 1;  PI(ptr) = 2*i-1;  PJ(ptr) = 2*ki-1;  PV(ptr) = omega;
                ptr = ptr + 1;  PI(ptr) = 2*i;    PJ(ptr) = 2*ki;    PV(ptr) = omega;
            end

        end
    end

    P = sparse(PI(1:ptr), PJ(1:ptr), PV(1:ptr), 2*N, 2*Nc);
end
