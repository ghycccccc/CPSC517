function hierarchy = amg_setup(A, theta, max_levels, coarse_threshold)
% AMG_SETUP  Build the AMG multilevel hierarchy (setup phase).
%
%   hierarchy = amg_setup(A, theta, max_levels, coarse_threshold)
%
% This function is called ONCE before GMRES. It produces the full
% multilevel hierarchy used by every subsequent preconditioner application.
%
% INPUTS:
%   A                 - fine-grid matrix (sparse, square)
%   theta             - strong dependence threshold, typically 0.25
%   max_levels        - maximum number of levels allowed
%   coarse_threshold  - stop coarsening when matrix size < this value
%
% OUTPUT:
%   hierarchy  - struct array; hierarchy{l} contains:
%                  .A  : operator at level l
%                  .P  : prolongation  (fine -> coarse), empty at coarsest
%                  .R  : restriction   (coarse -> fine), empty at coarsest
%
% THEORETICAL BASIS:
%   Each level is constructed by:
%     1. Computing strong dependence sets from matrix entries
%     2. C/F splitting via greedy coloring (+ second pass for H-1)
%     3. Building interpolation P from the AMG weight formula
%     4. Setting R = P^T  (variational property)
%     5. Galerkin coarse operator:  A_c = P^T * A * P

    hierarchy    = {};
    hierarchy{1} = struct('A', A, 'P', [], 'R', []);

    for lev = 1 : max_levels - 1

        A_lev = hierarchy{lev}.A;
        n     = size(A_lev, 1);

        % Stop if already small enough for a direct solve
        if n <= coarse_threshold
            break;
        end

        % ------------------------------------------------------------------
        % STEP 1: Strong dependence sets
        % ------------------------------------------------------------------
        [S, ST] = compute_strong_dependence(A_lev, theta);
        %   S{i}  = list of j such that i strongly depends on j
        %   ST{i} = list of j such that j strongly depends on i

        % ------------------------------------------------------------------
        % STEP 2: C/F splitting
        % ------------------------------------------------------------------
        [C_pts, F_pts] = cf_splitting(S, ST, n);
        %   C_pts : logical vector, true at coarse-grid points
        %   F_pts : logical vector, true at fine-only points

        % Check if coarsening made progress; abort if not
        if sum(C_pts) == 0 || sum(C_pts) == n
            break;
        end

        % ------------------------------------------------------------------
        % STEP 3: Build interpolation operator P  (n x n_coarse)
        % ------------------------------------------------------------------
        P = build_interpolation(A_lev, C_pts, F_pts, S, n);

        % ------------------------------------------------------------------
        % STEP 4: Restriction = transpose of interpolation
        % ------------------------------------------------------------------
        R = P';

        % ------------------------------------------------------------------
        % STEP 5: Galerkin coarse-grid operator
        % ------------------------------------------------------------------
        A_coarse = R * A_lev * P;

        % Store this level
        hierarchy{lev}.P   = P;
        hierarchy{lev}.R   = R;

        % Add the coarse level (will be refined further or used as coarsest)
        hierarchy{lev+1} = struct('A', A_coarse, 'P', [], 'R', []);

    end
end


% =========================================================================
%  LOCAL FUNCTION 1: Strong dependence sets
% =========================================================================
function [S, ST] = compute_strong_dependence(A, theta)
% For each row i, find columns j such that:
%   -a_{ij} >= theta * max_{k~=i} { -a_{ik} }
% Only negative off-diagonal entries qualify (positive entries are weak).
%
% S{i}  : indices j that strongly influence i  (i strongly depends on j)
% ST{i} : indices j that i strongly influences (j strongly depends on i)

    n  = size(A, 1);
    S  = cell(n, 1);
    ST = cell(n, 1);
    for k = 1:n
        ST{k} = [];
    end

    for i = 1:n
        row    = A(i, :);          % full row (sparse)
        offdiag_neg = -row;        % negate: strong connections have large positive value
        offdiag_neg(i) = 0;        % exclude diagonal

        max_val = max(offdiag_neg);

        if max_val <= 0
            % No negative off-diagonal entries -> no strong connections
            S{i} = [];
            continue;
        end

        % Find columns j where -a_{ij} >= theta * max_val
        strong_j = find(offdiag_neg >= theta * max_val);
        S{i}     = strong_j;

        % Populate transpose relation: i strongly influences each j in S{i}
        for j = strong_j(:)'
            ST{j} = [ST{j}, i];
        end
    end
end


% =========================================================================
%  LOCAL FUNCTION 2: C/F splitting (coloring + second pass)
% =========================================================================
function [C_pts, F_pts] = cf_splitting(S, ST, n)
% FIRST PASS: greedy coloring guided by influence scores lambda_i = |ST{i}|
%
% Algorithm:
%   1. Initialize score lambda_i = |ST{i}| for all i
%   2. Pick unassigned i* with maximum lambda -> assign to C
%   3. Assign all unassigned j in ST{i*} to F
%   4. For each new F-point j, increment lambda_k for unassigned k in S{j}
%   5. Repeat until all assigned
%
% SECOND PASS: enforce heuristic H-1
%   For each F-point i, every j in S{i} must either be in C_i (coarse
%   interpolatory set of i) or must strongly depend on some point in C_i.
%   If not, promote j to C.

    status = zeros(n, 1);   % 0=unassigned, 1=C-point, -1=F-point
    lambda = cellfun(@length, ST);   % initial scores

    % ---- First pass ----
    num_assigned = 0;
    while num_assigned < n

        % Find unassigned node with maximum lambda
        unassigned = find(status == 0);
        if isempty(unassigned), break; end

        [~, idx] = max(lambda(unassigned));
        i_star   = unassigned(idx);

        % Assign i_star to C
        status(i_star) = 1;
        num_assigned   = num_assigned + 1;

        % Assign all unassigned members of ST{i_star} to F
        for j = ST{i_star}(:)'
            if status(j) == 0
                status(j)    = -1;
                num_assigned = num_assigned + 1;

                % Increment scores of unassigned nodes that strongly influence j
                for k = S{j}(:)'
                    if status(k) == 0
                        lambda(k) = lambda(k) + 1;
                    end
                end
            end
        end

        % Any remaining unassigned neighbors of i_star that weren't in ST
        % will be picked up in subsequent iterations naturally.
    end

    % Any node still unassigned (isolated nodes) -> assign to C
    status(status == 0) = 1;

    % ---- Second pass: enforce H-1 ----
    % For each F-point i, check that every j in S{i} either:
    %   (a) is a C-point that directly interpolates to i, OR
    %   (b) is an F-point that strongly depends on at least one C-point
    %       also in S{i}
    % If neither holds, promote j to C.

    for i = find(status == -1)'
        C_i = intersect(S{i}, find(status == 1));   % current coarse interpolatory set

        for j = S{i}(:)'
            if status(j) == 1
                % j is already a C-point, H-1 satisfied for this j
                continue;
            end
            % j is an F-point strongly influencing i
            % Check if j strongly depends on at least one node in C_i
            common = intersect(S{j}, C_i);
            if isempty(common)
                % H-1 violated: promote j to C-point
                status(j) = 1;
                %disp(C_i)
                %disp(j)
                C_i       = [C_i; j];   %#ok<AGROW>
            end
        end
    end

    C_pts = (status ==  1);
    F_pts = (status == -1);
end


% =========================================================================
%  LOCAL FUNCTION 3: Build interpolation operator P
% =========================================================================
function P = build_interpolation(A, C_pts, F_pts, S, n)
% Construct the prolongation operator P of size n x n_coarse.
%
% For a C-point i: row i of P has a single 1 in the column corresponding
%   to i's coarse index.
%
% For an F-point i: row i of P has interpolation weights omega_{ij} for
%   each j in C_i (the coarse interpolatory set of i):
%
%   omega_{ij} = - [ a_{ij} + sum_{m in D^s_i} ( a_{im}*a_{mj} /
%                                                  sum_{k in C_i} a_{mk} ) ]
%                / [ a_{ii} + sum_{n in D^w_i} a_{in} ]
%
% where:
%   C_i   = strongly influencing C-point neighbors of i
%   D^s_i = strongly influencing F-point neighbors of i
%   D^w_i = weakly connected neighbors of i (any C or F)

    % Map from fine-grid index to coarse-grid column index
    coarse_indices          = zeros(n, 1);
    coarse_indices(C_pts)   = 1 : sum(C_pts);
    n_coarse                = sum(C_pts);

    % Build P via triplets
    PI = [];  PJ = [];  PV = [];

    for i = 1:n

        if C_pts(i)
            % ---- C-point: identity row ----
            PI = [PI, i];                       %#ok<AGROW>
            PJ = [PJ, coarse_indices(i)];       %#ok<AGROW>
            PV = [PV, 1.0];                     %#ok<AGROW>

        else
            % ---- F-point: compute interpolation weights ----

            % Partition neighbors of i into C_i, D^s_i, D^w_i
            Ni       = S{i};          % all strong influencers of i
            C_i      = Ni(C_pts(Ni));                    % strong C-neighbors
            Ds_i     = Ni(~C_pts(Ni));                   % strong F-neighbors

            % Weak neighbors: all j~=i with a_{ij}~=0, not in Ni
            row_i    = A(i, :);
            all_nbrs = find(row_i ~= 0);
            all_nbrs = all_nbrs(all_nbrs ~= i);
            Dw_i     = setdiff(all_nbrs, Ni);

            % Denominator: a_{ii} + sum_{n in D^w_i} a_{in}
            denom = A(i, i) + sum(A(i, Dw_i));

            if abs(denom) < 1e-14 || isempty(C_i)
                % Degenerate case: no coarse neighbors -> treat as isolated
                % Assign uniform weight over any C-point (fallback)
                if n_coarse > 0
                    % Find nearest C-point by index (rough fallback)
                    c_all = find(C_pts);
                    [~, closest] = min(abs(c_all - i));
                    PI = [PI, i];                           %#ok<AGROW>
                    PJ = [PJ, coarse_indices(c_all(closest))]; %#ok<AGROW>
                    PV = [PV, 1.0];                         %#ok<AGROW>
                end
                continue;
            end

            % For each j in C_i, compute omega_{ij}
            for j = C_i(:)'

                % Numerator base: a_{ij}
                num = A(i, j);

                % Add contributions from D^s_i nodes
                for m = Ds_i(:)'
                    sum_amk = sum(A(m, C_i));   % sum_{k in C_i} a_{mk}
                    if abs(sum_amk) > 1e-14
                        num = num + A(i, m) * A(m, j) / sum_amk;
                    end
                end

                omega_ij = -num / denom;

                PI = [PI, i];                        %#ok<AGROW>
                PJ = [PJ, coarse_indices(j)];        %#ok<AGROW>
                PV = [PV, omega_ij];                 %#ok<AGROW>
            end

        end % if C_pts(i)

    end % for i

    P = sparse(PI, PJ, PV, n, n_coarse);
end
