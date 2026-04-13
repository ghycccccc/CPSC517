% =========================================================================
%  FILE: solve_with_amg_gmres.m
%  Top-level driver: sets up AMG hierarchy and runs preconditioned GMRES.
% =========================================================================
function [x, resvec, iter] = solve_with_amg_gmres(K, b, amg_opts, gmres_opts)
% SOLVE_WITH_AMG_GMRES  Solve K*x = b using AMG-preconditioned GMRES.
%
%   [x, resvec, iter] = solve_with_amg_gmres(K, b, amg_opts, gmres_opts)
%
% INPUTS:
%   K          - system matrix (from build_navier_cauchy_heterogeneous)
%   b          - right-hand side vector
%   amg_opts   - struct with fields:
%                  .theta             (default 0.25)
%                  .max_levels        (default 10)
%                  .coarse_threshold  (default 50)
%                  .nu1               pre-smoothing sweeps  (default 1)
%                  .nu2               post-smoothing sweeps (default 1)
%   gmres_opts - struct with fields:
%                  .tol               (default 1e-8)
%                  .max_iter          (default 500)
%                  .restart           (default 30)
%
% OUTPUTS:
%   x       - solution vector
%   resvec  - relative residual norm at each GMRES iteration
%   iter    - number of GMRES iterations taken
%
% HOW THE PRECONDITIONER IS PASSED TO GMRES:
%   Matlab's gmres() signature is:
%     gmres(A, b, restart, tol, maxit, M1, M2, x0)
%   where M1 (and optionally M2) can be:
%     - a matrix M  (Matlab applies M\r internally at each step)
%     - a function handle @(r) M_inv_times_r
%   We use the function handle form because M_AMG^{-1} is never
%   explicitly formed as a matrix (it would be dense).
%
%   The function handle captures `hierarchy`, `nu1`, `nu2` via closure,
%   so GMRES only needs to call it with the current residual r.

    % ---- Parse AMG options with defaults ----
    if ~isfield(amg_opts, 'theta'),            amg_opts.theta            = 0.25; end
    if ~isfield(amg_opts, 'max_levels'),       amg_opts.max_levels       = 10;   end
    if ~isfield(amg_opts, 'coarse_threshold'), amg_opts.coarse_threshold = 50;   end
    if ~isfield(amg_opts, 'nu1'),              amg_opts.nu1              = 1;    end
    if ~isfield(amg_opts, 'nu2'),              amg_opts.nu2              = 1;    end

    % ---- Parse GMRES options with defaults ----
    if ~isfield(gmres_opts, 'tol'),      gmres_opts.tol      = 1e-8; end
    if ~isfield(gmres_opts, 'max_iter'), gmres_opts.max_iter = 500;  end
    if ~isfield(gmres_opts, 'restart'),  gmres_opts.restart  = 30;   end

    % ------------------------------------------------------------------
    % SETUP PHASE (called once)
    % Builds the full multilevel hierarchy: all coarse operators,
    % interpolation and restriction operators.
    % Cost: O(nnz(K) * num_levels), typically O(nnz(K)) since each
    % level has roughly half the unknowns.
    % ------------------------------------------------------------------
    fprintf('AMG Setup: building hierarchy...\n');
    t_setup   = tic;
    hierarchy = amg_setup(K, amg_opts.theta, ...
                          amg_opts.max_levels, ...
                          amg_opts.coarse_threshold);
    t_setup   = toc(t_setup);

    num_levels = length(hierarchy);
    fprintf('  Levels built: %d\n', num_levels);
    fprintf('  Finest grid:  %d unknowns\n', size(hierarchy{1}.A, 1));
    fprintf('  Coarsest grid: %d unknowns\n', size(hierarchy{end}.A, 1));
    fprintf('  Setup time: %.3f seconds\n', t_setup);

    % Print grid and operator complexity (as defined in Briggs Chapter 8)
    total_rows = 0;  total_nnz = 0;
    fine_rows  = size(hierarchy{1}.A, 1);
    fine_nnz   = nnz(hierarchy{1}.A);
    for l = 1:num_levels
        total_rows = total_rows + size(hierarchy{l}.A, 1);
        total_nnz  = total_nnz  + nnz(hierarchy{l}.A);
    end
    fprintf('  Grid complexity:     %.3f\n', total_rows / fine_rows);
    fprintf('  Operator complexity: %.3f\n', total_nnz  / fine_nnz);

    % ------------------------------------------------------------------
    % BUILD PRECONDITIONER FUNCTION HANDLE
    %
    % The function handle @(r) amg_preconditioner(hierarchy, r, nu1, nu2)
    % captures hierarchy, nu1, nu2 by closure. GMRES sees a simple
    % function of one argument: the current residual r.
    %
    % Each call to this handle:
    %   - applies exactly one V-cycle from zero initial guess
    %   - returns z ≈ K^{-1} r
    %   - is a fixed linear map (M^{-1} does not change between calls)
    %
    % GMRES calls this handle exactly once per iteration.
    % After m iterations, it has been called m times total.
    % ------------------------------------------------------------------
    nu1    = amg_opts.nu1;
    nu2    = amg_opts.nu2;
    M_func = @(r) amg_preconditioner(hierarchy, r, nu1, nu2);

    % ------------------------------------------------------------------
    % GMRES SOLVE PHASE
    %
    % Matlab gmres signature used here:
    %   [x, flag, relres, iter, resvec] = ...
    %       gmres(A, b, restart, tol, maxit, M1fun)
    %
    % M1fun is the LEFT preconditioner: GMRES minimizes
    %   || M1^{-1}(b - A x) ||_2
    % over the Krylov subspace, which changes the subspace built but
    % preserves the standard GMRES optimality property.
    %
    % iter is returned as [outer, inner]; total = (outer-1)*restart + inner
    % ------------------------------------------------------------------
    fprintf('GMRES Solve: running with AMG preconditioner...\n');
    t_solve = tic;

    [x, flag, relres, iter_out, resvec] = gmres( ...
        K,                       ... % system matrix
        b,                       ... % right-hand side
        gmres_opts.restart,      ... % restart parameter (Krylov subspace size)
        gmres_opts.tol,          ... % relative residual tolerance
        gmres_opts.max_iter,     ... % maximum iterations
        M_func                   ... % preconditioner function handle (left precond.)
    );

    t_solve = toc(t_solve);

    % Total iteration count from gmres output [outer_iters, inner_iters]
    if numel(iter_out) == 2
        iter = (iter_out(1) - 1) * gmres_opts.restart + iter_out(2);
    else
        iter = iter_out;
    end

    fprintf('  GMRES flag:      %d  (0=converged)\n', flag);
    fprintf('  Relative resid:  %.2e\n', relres);
    fprintf('  Iterations:      %d\n',   iter);
    fprintf('  Solve time:      %.3f seconds\n', t_solve);
    fprintf('  Setup/solve ratio: %.1f iterations equivalent\n', ...
            t_setup / (t_solve / max(iter,1)));

end
