% =========================================================================
%  demo_lognormal_permeability.m
%
%  Compares plain GMRES, ILU(0), Geometric MG, and AMG on a 2D scalar
%  diffusion problem with random log-normal permeability:
%
%     -div( kappa(x,y) grad u ) = 1   on [0,1]^2,   u = 0 on boundary
%
%     kappa(x,y) = exp( sigma * G(x,y) )
%
%  where G is a spatially correlated Gaussian field (correlation length
%  l_c).  High sigma creates an irregular high-permeability backbone
%  that a fixed Cartesian hierarchy cannot track; AMG's algebraic
%  coarsening follows the backbone automatically.
%
%  Requires:
%    make_diffusion_benchmark_case.m
%    build_scalar_diffusion_2d.m
%    run_preconditioner_experiment.m
%    amg_setup.m
%    amg_vcycle_and_gmres.m  (or amg_vcycle.m + amg_preconditioner.m)
%    geomg_setup.m
%    geomg_preconditioner.m
% =========================================================================
clear; clc; close all;

% -------------------------------------------------------------------------
% 1.  PARAMETERS
% -------------------------------------------------------------------------
nx = 60;   % interior nodes in x  ->  n = 6400 DOFs
ny = 60;   % interior nodes in y

sigma_ln = 2.5;   % log-standard-deviation of kappa
                  % 1-sigma contrast ~ e^sigma = 12x
                  % 2-sigma contrast ~ e^(2*sigma) = 148x
corr_len = 0.12;  % spatial correlation length (fraction of unit domain)
rng_seed = 42;    % fix seed so results are reproducible

% -------------------------------------------------------------------------
% 2.  BUILD THE COEFFICIENT FIELD AND ASSEMBLE THE SYSTEM
% -------------------------------------------------------------------------
fprintf('=========================================\n');
fprintf('  Log-normal Permeability Benchmark\n');
fprintf('=========================================\n\n');

case_opts = struct();
case_opts.sigma_ln  = sigma_ln;
case_opts.corr_len  = corr_len;
case_opts.rng_seed  = rng_seed;
case_opts.forcing   = 'ones';
case_opts.bc_variant = 'dirichlet_all';

case_data = make_diffusion_benchmark_case('lognormal', nx, ny, case_opts);

[A, b, grid_meta] = build_scalar_diffusion_2d( ...
    nx, ny, case_data.a, case_data.f, case_data.bc, ...
    struct('face_average', 'harmonic'));

fprintf('Grid:       %d x %d  (%d DOFs)\n',   nx, ny, size(A, 1));
fprintf('sigma:      %.1f\n',                  sigma_ln);
fprintf('Contrast:   ~%dx  (max/min kappa)\n', case_data.case_meta.contrast);
fprintf('nnz(A):     %d\n\n',                  nnz(A));

% -------------------------------------------------------------------------
% 3.  PROBLEM STRUCT
%     coeff_field carries log10(kappa) for a perceptually uniform display
%     of the many-orders-of-magnitude variation.
% -------------------------------------------------------------------------
problem             = struct();
problem.A           = A;
problem.b           = b;
problem.name        = case_data.name;
problem.grid_meta   = grid_meta;
problem.case_meta   = case_data.case_meta;
problem.coeff_field = log10(case_data.a);        % visualized on log scale
problem.coeff_label = 'log_{10} \kappa(x,y)';   % TeX string for title

% -------------------------------------------------------------------------
% 4.  SOLVER OPTIONS
% -------------------------------------------------------------------------
gmres_opts.tol      = 1e-8;
gmres_opts.max_iter = 300;
gmres_opts.restart  = 50;

% AMG: two pre/post-smoothing sweeps improve robustness at high contrast
amg_opts.theta            = 0.25;
amg_opts.max_levels       = 8;
amg_opts.coarse_threshold = 60;
amg_opts.nu1              = 2;
amg_opts.nu2              = 2;

% ILU: zero-fill (nofill) — the standard cheap baseline
pc_opts.methods   = {'plain', 'ilu', 'geomg', 'amg'};
pc_opts.ilu_setup = struct('type', 'nofill');

% -------------------------------------------------------------------------
% 5.  RUN EXPERIMENT
%     Two figures open automatically:
%       "Problem Setup"            — kappa field + matrix colormap
%       "Preconditioner Convergence" — log-log residual histories
% -------------------------------------------------------------------------
results = run_preconditioner_experiment(problem, gmres_opts, amg_opts, pc_opts);

% -------------------------------------------------------------------------
% 6.  SIGMA SWEEP  (optional — comment out if only one run is needed)
%     Shows how iteration counts scale with log-contrast for each method.
% -------------------------------------------------------------------------
run_sigma_sweep = true;

if run_sigma_sweep
    sigma_vals = [1.0, 3.0];%[1.0, 1.5, 2.0, 2.5, 3.0];
    methods_sw = {'plain', 'ilu', 'geomg', 'amg'};

    iter_table = zeros(length(sigma_vals), length(methods_sw));
    resid_table = zeros(length(sigma_vals), length(methods_sw));

    fprintf('\n=========================================\n');
    fprintf('  Sigma sweep (fixed seed=%d)\n', rng_seed);
    fprintf('=========================================\n');
    fprintf('%-8s', 'sigma');
    for m = 1:length(methods_sw)
        fprintf('%-14s', upper(methods_sw{m}));
    end
    fprintf('\n');

    sweep_amg  = amg_opts;
    sweep_gopt = gmres_opts;
    sweep_pc   = pc_opts;
    sweep_pc.methods = methods_sw;

    for si_idx = 1:length(sigma_vals)
        s = sigma_vals(si_idx);
        co = case_opts;
        co.sigma_ln = s;
        cd_sw = make_diffusion_benchmark_case('lognormal', nx, ny, co);
        [A_sw, b_sw, gm_sw] = build_scalar_diffusion_2d( ...
            nx, ny, cd_sw.a, cd_sw.f, cd_sw.bc, ...
            struct('face_average', 'harmonic'));

        prob_sw            = struct();
        prob_sw.A          = A_sw;
        prob_sw.b          = b_sw;
        prob_sw.name       = cd_sw.name;
        prob_sw.grid_meta  = gm_sw;
        prob_sw.case_meta  = cd_sw.case_meta;
        % Skip coeff_field/coeff_label for sweep runs to avoid extra figures
        res_sw = run_preconditioner_experiment( ...
            prob_sw, sweep_gopt, sweep_amg, sweep_pc);
        close(findobj('Name', 'Problem Setup'));
        close(findobj('Name', 'Preconditioner Convergence'));

        fprintf('%-8.1f', s);
        for m = 1:length(methods_sw)
            iter_table(si_idx, m)  = res_sw(m).iter;
            resid_table(si_idx, m) = res_sw(m).true_resid;
            fmt = '%6d(%.0e)  ';
            fprintf(fmt, res_sw(m).iter, res_sw(m).true_resid);
        end
        fprintf('\n');
    end

    % ---- Plot: iteration count vs sigma ----
    colors_sw = {[0 0 0], [0.85 0.1 0.1], [0.85 0.45 0.10], [0 0.25 0.9]};
    styles_sw = {'--',    '-',             '-',               '--'  };
    widths_sw = { 2.0,     1.8,             1.8,               2.8  };

    figure('Name', 'Sigma Sweep', 'Position', [120 600 640 400]);
    for m = 1:length(methods_sw)
        semilogy(sigma_vals, iter_table(:, m), styles_sw{m}, ...
            'Color', colors_sw{m}, 'LineWidth', widths_sw{m}, ...
            'DisplayName', upper(methods_sw{m}), 'Marker', 'o', ...
            'MarkerSize', 6, 'MarkerFaceColor', colors_sw{m});
        hold on;
    end
    grid on;
    xlabel('\sigma  (log-standard-deviation of \kappa)');
    ylabel('GMRES iterations to convergence');
    title('Iteration count vs. log-contrast  (log-normal permeability)');
    legend('Location', 'northwest');
    set(gca, 'YScale', 'log');

    % Save sigma-sweep summary alongside the per-run figures
    fig_dir_demo = fullfile(fileparts(mfilename('fullpath')), '..', 'figures');
    if ~exist(fig_dir_demo, 'dir'), mkdir(fig_dir_demo); end
    print(gcf, fullfile(fig_dir_demo, 'lognormal_sigma_sweep.png'), '-dpng', '-r150');
    fprintf('  [saved] lognormal_sigma_sweep.png\n');
end
