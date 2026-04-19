function [x, resvec, iter] = solve_with_block_amg_gmres(K, b, amg_opts, gmres_opts)
% SOLVE_WITH_BLOCK_AMG_GMRES  Block AMG-preconditioned GMRES driver.
%
%   Builds a block AMG hierarchy via block_amg_setup, then uses one
%   block V-cycle per GMRES iteration as the right preconditioner.
%   Mirrors the interface of solve_with_amg_gmres.m for easy comparison.
%
% INPUTS:
%   K         - sparse [2N x 2N] system matrix (interleaved DOF ordering)
%   b         - RHS vector [2N x 1]
%   amg_opts  - struct with fields:
%                 .theta            (default 0.25)
%                 .max_levels       (default 10)
%                 .coarse_threshold (default 40,  in nodes)
%                 .nu1              (default 1,   pre-smooth sweeps)
%                 .nu2              (default 1,   post-smooth sweeps)
%   gmres_opts - struct with fields:
%                 .tol      (default 1e-8)
%                 .max_iter (default 500)
%                 .restart  (default 30)
%
% OUTPUTS:
%   x       - approximate solution
%   resvec  - residual norm history (length = iterations + 1)
%   iter    - total GMRES iterations performed

    if ~isfield(amg_opts, 'theta'),            amg_opts.theta            = 0.25; end
    if ~isfield(amg_opts, 'max_levels'),       amg_opts.max_levels       = 10;   end
    if ~isfield(amg_opts, 'coarse_threshold'), amg_opts.coarse_threshold = 40;   end
    if ~isfield(amg_opts, 'nu1'),              amg_opts.nu1              = 1;    end
    if ~isfield(amg_opts, 'nu2'),              amg_opts.nu2              = 1;    end

    if ~isfield(gmres_opts, 'tol'),      gmres_opts.tol      = 1e-32; end
    if ~isfield(gmres_opts, 'max_iter'), gmres_opts.max_iter = 500;  end
    if ~isfield(gmres_opts, 'restart'),  gmres_opts.restart  = 30;   end

    % ------------------------------------------------------------------
    % AMG setup
    % ------------------------------------------------------------------
    fprintf('Block AMG Setup: building hierarchy...\n');
    t_setup   = tic;
    hierarchy = block_amg_setup(K, amg_opts.theta, ...
                                amg_opts.max_levels, amg_opts.coarse_threshold);
    t_setup   = toc(t_setup);

    num_levels = length(hierarchy);
    fprintf('  Levels:        %d\n', num_levels);
    fprintf('  Finest size:   %d  (%d nodes)\n', ...
            size(hierarchy{1}.A, 1), size(hierarchy{1}.A, 1)/2);
    fprintf('  Coarsest size: %d  (%d nodes)\n', ...
            size(hierarchy{end}.A, 1), size(hierarchy{end}.A, 1)/2);
    fprintf('  Setup time:    %.3f s\n', t_setup);

    fine_rows = size(hierarchy{1}.A, 1);
    fine_nnz  = nnz(hierarchy{1}.A);
    tot_rows  = 0;  tot_nnz = 0;
    for l = 1 : num_levels
        tot_rows = tot_rows + size(hierarchy{l}.A, 1);
        tot_nnz  = tot_nnz  + nnz(hierarchy{l}.A);
    end
    fprintf('  Grid complexity:     %.3f\n', tot_rows / fine_rows);
    fprintf('  Operator complexity: %.3f\n', tot_nnz  / fine_nnz);

    % ------------------------------------------------------------------
    % Preconditioner: one block V-cycle from zero (linear operator)
    % ------------------------------------------------------------------
    nu1    = amg_opts.nu1;
    nu2    = amg_opts.nu2;
    M_func = @(r) block_amg_vcycle(hierarchy, 1, r, zeros(size(r)), nu1, nu2);

    % ------------------------------------------------------------------
    % GMRES solve
    % ------------------------------------------------------------------
    fprintf('GMRES Solve...\n');
    t_solve = tic;

    [x, flag, relres, iter_out, resvec] = gmres( ...
        K, b, gmres_opts.restart, gmres_opts.tol, gmres_opts.max_iter, M_func);

    t_solve = toc(t_solve);

    % Compute total iteration count (same correction as solve_with_amg_gmres.m)
    if numel(iter_out) == 2
        outer = iter_out(1);
        inner = iter_out(2);
        if inner == 0
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
                t_setup / (t_solve / iter));
    end

end
