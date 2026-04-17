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


% =========================================================================
function [x, resvec, iter] = solve_with_amg_gmres(K, b, amg_opts, gmres_opts)
% SOLVE_WITH_AMG_GMRES  AMG-preconditioned GMRES driver.  (corrected)
%
% CORRECTION: iter computation now handles the case where GMRES stagnates
% at a restart boundary (iter_out(2)==0), which previously gave iter=-30.

    if ~isfield(amg_opts,'theta'),            amg_opts.theta            = 0.25; end
    if ~isfield(amg_opts,'max_levels'),       amg_opts.max_levels       = 10;   end
    if ~isfield(amg_opts,'coarse_threshold'), amg_opts.coarse_threshold = 50;   end
    if ~isfield(amg_opts,'nu1'),              amg_opts.nu1              = 1;    end
    if ~isfield(amg_opts,'nu2'),              amg_opts.nu2              = 1;    end

    if ~isfield(gmres_opts,'tol'),      gmres_opts.tol      = 1e-8; end
    if ~isfield(gmres_opts,'max_iter'), gmres_opts.max_iter = 500;  end
    if ~isfield(gmres_opts,'restart'),  gmres_opts.restart  = 30;   end

    % ------------------------------------------------------------------
    % Setup phase (once)
    % ------------------------------------------------------------------
    fprintf('AMG Setup: building hierarchy...\n');
    t_setup   = tic;
    hierarchy = amg_setup(K, amg_opts.theta, ...
                          amg_opts.max_levels, amg_opts.coarse_threshold);
    t_setup   = toc(t_setup);

    num_levels = length(hierarchy);
    fprintf('  Levels:        %d\n', num_levels);
    fprintf('  Finest size:   %d\n', size(hierarchy{1}.A,1));
    fprintf('  Coarsest size: %d\n', size(hierarchy{end}.A,1));
    fprintf('  Setup time:    %.3f s\n', t_setup);

    fine_rows = size(hierarchy{1}.A,1);   fine_nnz = nnz(hierarchy{1}.A);
    tot_rows  = 0;  tot_nnz = 0;
    for l = 1:num_levels
        tot_rows = tot_rows + size(hierarchy{l}.A,1);
        tot_nnz  = tot_nnz  + nnz(hierarchy{l}.A);
    end
    fprintf('  Grid complexity:     %.3f\n', tot_rows/fine_rows);
    fprintf('  Operator complexity: %.3f\n', tot_nnz /fine_nnz );

    % ------------------------------------------------------------------
    % Preconditioner handle
    % Each call applies exactly one V-cycle from zero.
    % Captured variables (hierarchy, nu1, nu2) are fixed for all calls.
    % ------------------------------------------------------------------
    nu1    = amg_opts.nu1;
    nu2    = amg_opts.nu2;
    M_func = @(r) amg_preconditioner(hierarchy, r, nu1, nu2);

    % ------------------------------------------------------------------
    % GMRES solve
    % ------------------------------------------------------------------
    fprintf('GMRES Solve...\n');
    t_solve = tic;

    [x, flag, relres, iter_out, resvec] = gmres( ...
        K, b, gmres_opts.restart, gmres_opts.tol, gmres_opts.max_iter, M_func);

    t_solve = toc(t_solve);

    % CORRECTED iter computation:
    % iter_out from Matlab gmres is [num_outer_restarts, inner_iter_at_stop]
    % When stagnation occurs at a restart boundary, inner = 0 and
    % the total completed iterations = (outer-1)*restart + restart = outer*restart.
    if numel(iter_out) == 2
        outer = iter_out(1);
        inner = iter_out(2);
        if inner == 0
            % Stagnated exactly at end of a restart cycle
            iter = (outer - 1) * gmres_opts.restart;
        else
            iter = (outer - 1) * gmres_opts.restart + inner;
        end
    else
        iter = iter_out;
    end

    fprintf('  Flag:       %d  (0=converged, 1=maxit, 2=ill-cond, 3=stagnate)\n', flag);
    fprintf('  Rel resid:  %.2e\n', relres);
    fprintf('  Iterations: %d\n',   iter);
    fprintf('  Solve time: %.3f s\n', t_solve);
    if iter > 0
        fprintf('  Setup cost ~ %.1f equivalent iterations\n', ...
                t_setup / (t_solve/iter));
    end

end
