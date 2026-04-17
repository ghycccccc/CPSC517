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