function hierarchy = amg_setup(A, theta, max_levels, coarse_threshold)
% AMG_SETUP  Build AMG multilevel hierarchy.  (corrected)
%
%   KEY FIX: compute_strong_dependence now correctly handles matrices
%   with mixed-sign off-diagonal entries (as in heterogeneous elasticity).
%   Only genuinely negative off-diagonal entries qualify as strong;
%   threshold is applied to their magnitudes.
%
% INPUTS:
%   A                 - fine-grid matrix (sparse)
%   theta             - strong dependence threshold (default 0.25)
%   max_levels        - max hierarchy depth
%   coarse_threshold  - direct solve when size < this
%
% OUTPUT:
%   hierarchy{l}.A  : matrix at level l
%   hierarchy{l}.P  : prolongation (empty at coarsest)
%   hierarchy{l}.R  : restriction  (empty at coarsest)
    hierarchy    = {};
    hierarchy{1} = struct('A', A, 'P', [], 'R', []);

    for lev = 1 : max_levels - 1
        A_lev = hierarchy{lev}.A;
        n     = size(A_lev, 1);

        if n <= coarse_threshold, break; end

        [S, ST] = compute_strong_dependence(A_lev, theta);

        [C_pts, F_pts] = cf_splitting(S, ST, n);

        if sum(C_pts) == 0 || sum(C_pts) == n, break; end

        P = build_interpolation(A_lev, C_pts, F_pts, S, n);

        if isempty(P) || size(P,2) == 0, break; end

        R        = P';
        A_coarse = R * A_lev * P;

        % After: A_coarse = R * A_lev * P;
        n_fine   = size(A_lev, 1);
        n_coarse = size(A_coarse, 1);
        ratio    = n_coarse / n_fine;
        d        = diag(A_coarse);
        rowsums  = sum(abs(A_coarse), 2) - abs(d);
        dd_ratio = min(abs(d) ./ max(rowsums, 1e-14));  % <1 means NOT diag dominant
        cond_est = condest(A_coarse);                    % condition number estimate
        
        fprintf('  Level %d -> %d:  coarsen=%.2f  min_dd=%.3f  condest=%.2e\n', ...
                lev, lev+1, ratio, dd_ratio, cond_est);
        
        % Check interpolation weight range
        P_vals = nonzeros(P);
        fprintf('    P weights: min=%.3f  max=%.3f  n_negative=%d\n', ...
        min(P_vals), max(P_vals), sum(P_vals < 0));

        hierarchy{lev}.P   = P;
        hierarchy{lev}.R   = R;
        hierarchy{lev+1}   = struct('A', A_coarse, 'P', [], 'R', []);

    end
end


% =========================================================================
function [S, ST] = compute_strong_dependence(A, theta)
% Compute strong dependence sets for a possibly non-M-matrix.
%
% CORRECTION over previous version:
%   Previous code negated the entire row and took max(), which mishandled
%   positive off-diagonal entries (common in heterogeneous elasticity).
%
% Correct algorithm:
%   1. Extract off-diagonal entries of row i.
%   2. Consider ONLY entries that are strictly negative (genuine coupling).
%   3. max_neg = max magnitude among those negative entries.
%   4. j is a strong connection if  -a_{ij} >= theta * max_neg.
%   Positive off-diagonal entries are always treated as weak.

    n  = size(A, 1);
    S  = cell(n, 1);
    ST = cell(n, 1);
    for k = 1:n, ST{k} = []; end

    for i = 1:n
        % Get full row as dense vector for indexing convenience
        row_i = full(A(i,:));
        row_i(i) = 0;                    % exclude diagonal

        % Identify negative off-diagonal entries only
        neg_mask = (row_i < 0);
        if ~any(neg_mask)
            S{i} = [];
            continue;
        end

        neg_vals = -row_i(neg_mask);     % positive magnitudes
        max_neg  = max(neg_vals);

        if max_neg <= 0
            S{i} = [];
            continue;
        end

        % Strong if magnitude >= theta * max_magnitude
        all_cols = 1:n;
        strong_j = all_cols(neg_mask & (-row_i >= theta * max_neg));
        S{i}     = strong_j;

        for j = strong_j
            ST{j} = [ST{j}, i];
        end
    end
end


% =========================================================================
function [C_pts, F_pts] = cf_splitting(S, ST, n)
% C/F splitting: greedy coloring (first pass) + H-1 enforcement (second pass).
% Unchanged from previous version — logic was correct.

    status = zeros(n, 1);   % 0=unassigned, 1=C, -1=F
    lambda = cellfun(@length, ST);

    % ---- First pass: greedy ----
    num_assigned = 0;
    while num_assigned < n
        unassigned = find(status == 0);
        if isempty(unassigned), break; end

        [~, idx] = max(lambda(unassigned));
        i_star   = unassigned(idx);

        status(i_star) = 1;
        num_assigned   = num_assigned + 1;

        for j = ST{i_star}(:)'
            if status(j) == 0
                status(j)    = -1;
                num_assigned = num_assigned + 1;
                for k = S{j}(:)'
                    if status(k) == 0
                        lambda(k) = lambda(k) + 1;
                    end
                end
            end
        end
    end

    % Isolated (no strong connections) -> C
    status(status == 0) = 1;

    % ---- Second pass: enforce H-1 ----
    % For each F-point i, every j in S{i} must be a C-point
    % or strongly depend on some C-point in C_i.
    for i = find(status == -1)'
        C_i = intersect(S{i}, find(status == 1));

        for j = S{i}(:)'
            if status(j) == 1, continue; end   % already C, ok

            % j is F: check if j has at least one strong C-neighbor in C_i
            if isempty(intersect(S{j}, C_i))
                % H-1 violated: promote j to C
                status(j) = 1;
                C_i       = [C_i, j]; 
            end
        end
    end

    C_pts = (status ==  1);
    F_pts = (status == -1);
end


% =========================================================================
function P = build_interpolation(A, C_pts, F_pts, S, n)
% Build prolongation operator P.
%
% CORRECTION: removed fallback that assigned weight-1 to an arbitrary
% C-point for degenerate F-points with no C-neighbors. Such nodes are
% now promoted to C-points during the second pass of cf_splitting instead.
% If they still appear here (due to edge cases), they are added as C-points.

    coarse_indices        = zeros(n, 1);
    coarse_indices(C_pts) = 1 : sum(C_pts);
    n_coarse              = sum(C_pts);

    if n_coarse == 0
        P = sparse(n, 0);
        return;
    end

    PI = [];  PJ = [];  PV = [];

    for i = 1:n

        if C_pts(i)
            % C-point: identity row
            PI = [PI, i];                     
            PJ = [PJ, coarse_indices(i)];     
            PV = [PV, 1.0];                   

        else
            % F-point: compute AMG interpolation weights
            Ni   = S{i};
            C_i  = Ni(C_pts(Ni));    % strong C-neighbors
            Ds_i = Ni(~C_pts(Ni));   % strong F-neighbors

            % Weak neighbors: nonzero off-diagonal, not in S{i}
            row_i    = A(i,:);
            all_nbrs = find(row_i ~= 0);
            all_nbrs = all_nbrs(all_nbrs ~= i);
            Dw_i     = setdiff(all_nbrs, Ni);

            if isempty(C_i)
                % No strong C-neighbor: this node should have been promoted
                % to C during second pass. Treat it as C here as fallback.
                % This avoids the previous arbitrary nearest-C assignment.
                C_pts(i)          = true;
                n_coarse          = n_coarse + 1;
                coarse_indices(i) = n_coarse;
                PI = [PI, i];                     
                PJ = [PJ, coarse_indices(i)];     
                PV = [PV, 1.0];                   
                continue;
            end

            % Denominator: a_{ii} + sum of weak-neighbor entries
            denom = A(i,i) + sum(A(i, Dw_i));

            if abs(denom) < 1e-14
                % Degenerate denominator: treat as C-point
                C_pts(i)          = true;
                n_coarse          = n_coarse + 1;
                coarse_indices(i) = n_coarse;
                PI = [PI, i];                     
                PJ = [PJ, coarse_indices(i)];     
                PV = [PV, 1.0];                   
                continue;
            end

            % Compute omega_{ij} for each j in C_i
            for j = C_i(:)'
                num = A(i, j);

                % Contributions from strong F-neighbors via indirect coupling
                for m = Ds_i(:)'
                    sum_amk = sum(A(m, C_i));
                    if abs(sum_amk) > 1e-14
                        num = num + A(i,m) * A(m,j) / sum_amk;
                    end
                end

                omega_ij = -num / denom;

                PI = [PI, i];                   
                PJ = [PJ, coarse_indices(j)];   
                PV = [PV, omega_ij];            
            end

        end
    end

    P = sparse(PI, PJ, PV, n, n_coarse);
end
