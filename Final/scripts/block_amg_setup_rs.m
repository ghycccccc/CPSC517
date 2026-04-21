function hierarchy = block_amg_setup_rs(K, theta, max_levels, coarse_threshold)
% BLOCK_AMG_SETUP_RS  Block (2×2) AMG hierarchy with Ruge-Stüben block weights.
%
%   Identical to block_amg_setup in everything except the prolongation:
%
%   Prolongation — Direct RS block interpolation (no SA smoothing):
%     A generalization of classical scalar Ruge-Stüben interpolation to
%     2×2 blocks.  The weight ω_ij for F-node i interpolating from C-node j
%     is a full 2×2 matrix:
%
%       ω_ij = D_inv * ( A_ij  +  T_ij  +  E_ij )
%
%     where the three numerator terms are:
%
%       Direct coupling:
%         A_ij  — the actual 2×2 off-diagonal block
%
%       Twice-removed (T_i = strong F-nbrs of i that share ≥1 C-nbr with i):
%         T_ij  = Σ_{k ∈ T_i} A_ik * ‖A_kj‖_F / Σ_{l ∈ I_i ∩ I_k} ‖A_kl‖_F
%                 (distributes indirect coupling proportionally by Frobenius norm)
%
%       Simple average (E_i = strong F-nbrs of i with no common C-nbr):
%         E_ij  = (1/|I_i|) * Σ_{k ∈ E_i} A_ik
%                 (equal share of isolated F-nbr coupling to each C-nbr)
%
%     Denominator (2×2, inverted):
%         D     = A_block(i,i) + Σ_{k ∈ W_i} A_block(i,k)
%                 W_i = weak neighbors (nonzero block coupling, not strongly connected)
%
%   C-nodes receive the exact identity block I_2 (injection).
%
%   This approach satisfies the AMG interpolation property for algebraically
%   smooth errors directly through the local equilibrium equation, without
%   requiring a subsequent SA-style Jacobi prolongation smoothing step.
%
%   The resulting ω_ij blocks are 2×2 matrices (not scalar × I_2), so P
%   can represent anisotropic coupling between u and v DOFs of an F-node.
%
% INPUTS / OUTPUTS: identical to block_amg_setup.

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

        % Node-level strength matrix and C/F split (identical to block_amg_setup)
        S_nodes               = node_strength_matrix(A_lev, N);
        [S_strong, ST_strong] = node_strong_dependence(S_nodes, N, theta);
        [C_nodes, ~]          = cf_splitting(S_strong, ST_strong, N);

        if sum(C_nodes) == 0 || sum(C_nodes) == N, break; end

        % RS block prolongation — no SA smoothing step
        P = block_prolongation_rs(A_lev, S_nodes, C_nodes, S_strong, N);

        if isempty(P) || size(P, 2) == 0, break; end

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
% N×N node-level strength matrix; S(i,j) = Frobenius norm of 2×2 block A(2i-1:2i, 2j-1:2j).
    [AI, AJ, AV] = find(A);
    ni  = ceil(AI / 2);
    nj  = ceil(AJ / 2);
    off = (ni ~= nj);
    ni  = ni(off);   nj = nj(off);   av2 = AV(off).^2;
    S_sq = sparse(ni, nj, av2, N, N);
    [si, sj, sv] = find(S_sq);
    S = sparse(si, sj, sqrt(sv), N, N);
end


% =========================================================================
function [S_strong, ST_strong] = node_strong_dependence(S_nodes, N, theta)
% Strong-connection sets on the N-node graph (all Frobenius norms >= 0).
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
% Greedy + H-1 C/F splitting on the node graph (identical to block_amg_setup).
    status = zeros(n, 1);
    lambda = cellfun(@length, ST);

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
                    if status(k) == 0, lambda(k) = lambda(k) + 1; end
                end
            end
        end
    end
    status(status == 0) = 1;   % isolated nodes -> C

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
function P = block_prolongation_rs(A, S_nodes, C_nodes, S_strong, N)
% Build [2N × 2*Nc] RS block prolongation.
%
%   C-node i:  P(2i-1:2i, 2ki-1:2ki) = I_2  (exact injection)
%
%   F-node i:  ω_ij = D_inv * (A_ij + T_ij + E_per_j)   [2×2 matrix]
%     D     = A_block(i,i) + Σ_{k∈W_i} A_block(i,k)        [2×2, inverted]
%     T_ij  = Σ_{k∈T_i} A_ik * ‖A_kj‖_F / Σ_{l∈I_i∩I_k} ‖A_kl‖_F
%     E_per_j = (1/|I_i|) * Σ_{k∈E_i} A_ik

    Nc = sum(C_nodes);
    coarse_idx          = zeros(N, 1);
    coarse_idx(C_nodes) = 1 : Nc;

    % Pre-allocate triplet arrays (4 entries per 2×2 block weight per C-nbr)
    max_ent = 4 * N * 32;
    PI = zeros(max_ent, 1);
    PJ = zeros(max_ent, 1);
    PV = zeros(max_ent, 1);
    ptr = 0;

    for i = 1:N
        ri = 2*i-1 : 2*i;

        % ---- C-point: identity block ----
        if C_nodes(i)
            ki  = coarse_idx(i);
            ptr = ptr + 1;  PI(ptr) = 2*i-1;  PJ(ptr) = 2*ki-1;  PV(ptr) = 1.0;
            ptr = ptr + 1;  PI(ptr) = 2*i;    PJ(ptr) = 2*ki;    PV(ptr) = 1.0;
            continue;
        end

        % ---- F-point ----
        Ni   = S_strong{i};
        C_i  = Ni(C_nodes(Ni));    % strong C-neighbors  (interpolatory set I_i)
        Ds_i = Ni(~C_nodes(Ni));   % strong F-neighbors

        % Fallback: no strong C-neighbors -> promote to C
        if isempty(C_i)
            C_nodes(i)    = true;
            Nc            = Nc + 1;
            coarse_idx(i) = Nc;
            ki  = Nc;
            ptr = ptr + 1;  PI(ptr) = 2*i-1;  PJ(ptr) = 2*ki-1;  PV(ptr) = 1.0;
            ptr = ptr + 1;  PI(ptr) = 2*i;    PJ(ptr) = 2*ki;    PV(ptr) = 1.0;
            continue;
        end

        % ---- Denominator: diagonal block + weak neighbor blocks ----
        all_nbrs = find(S_nodes(i, :) > 0);
        all_nbrs = all_nbrs(all_nbrs ~= i);
        W_i   = setdiff(all_nbrs, Ni);       % nonzero coupling but not strong

        D_blk = full(A(ri, ri));
        for k = W_i(:)'
            rk    = 2*k-1 : 2*k;
            D_blk = D_blk + full(A(ri, rk));
        end

        det_D = D_blk(1,1)*D_blk(2,2) - D_blk(1,2)*D_blk(2,1);
        if abs(det_D) < 1e-14
            % Degenerate denominator -> promote to C
            C_nodes(i)    = true;
            Nc            = Nc + 1;
            coarse_idx(i) = Nc;
            ki  = Nc;
            ptr = ptr + 1;  PI(ptr) = 2*i-1;  PJ(ptr) = 2*ki-1;  PV(ptr) = 1.0;
            ptr = ptr + 1;  PI(ptr) = 2*i;    PJ(ptr) = 2*ki;    PV(ptr) = 1.0;
            continue;
        end
        D_inv = [D_blk(2,2), -D_blk(1,2); -D_blk(2,1), D_blk(1,1)] / det_D;

        % ---- Classify strong F-neighbors into T_i and E_i ----
        T_i = [];
        E_i = [];
        for k = Ds_i(:)'
            Nk  = S_strong{k};
            C_k = Nk(C_nodes(Nk));
            if ~isempty(intersect(C_i, C_k))
                T_i = [T_i, k];
            else
                E_i = [E_i, k];
            end
        end

        % ---- Pre-compute E_i contribution (same 2×2 added to each j∈C_i) ----
        n_Ci  = length(C_i);
        E_sum = zeros(2, 2);
        for k = E_i(:)'
            rk    = 2*k-1 : 2*k;
            E_sum = E_sum + full(A(ri, rk));
        end
        E_per_j = E_sum / n_Ci;    % 2×2; equal share for each coarse neighbor

        % ---- Build ω_ij for each j ∈ C_i ----
        for jj = 1 : n_Ci
            j  = C_i(jj);
            rj = 2*j-1 : 2*j;
            kj = coarse_idx(j);

            % Direct coupling block
            num = full(A(ri, rj));   % 2×2

            % Twice-removed contributions from T_i
            for k = T_i(:)'
                rk  = 2*k-1 : 2*k;
                Nk  = S_strong{k};
                C_k = Nk(C_nodes(Nk));
                common = intersect(C_i, C_k);   % I_i ∩ I_k

                % Scalar denominator: Σ_{l ∈ I_i∩I_k} ‖A_block(k,l)‖_F
                denom_k = 0;
                for l = common(:)'
                    rl      = 2*l-1 : 2*l;
                    denom_k = denom_k + norm(full(A(rk, rl)), 'fro');
                end
                if denom_k < 1e-14, continue; end

                A_kj_fro = norm(full(A(rk, rj)), 'fro');
                num = num + full(A(ri, rk)) * (A_kj_fro / denom_k);
            end

            % Simple average from E_i
            num = num + E_per_j;

            % 2×2 weight matrix
            omega = D_inv * num;

            % Insert as 2×2 block into P
            for r = 1:2
                for c = 1:2
                    ptr = ptr + 1;
                    PI(ptr) = 2*i + r - 2;
                    PJ(ptr) = 2*kj + c - 2;
                    PV(ptr) = omega(r, c);
                end
            end
        end
    end

    P = sparse(PI(1:ptr), PJ(1:ptr), PV(1:ptr), 2*N, 2*Nc);
end
