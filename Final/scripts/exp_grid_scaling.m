% =========================================================================
%  exp_grid_scaling.m
%
%  Experiment 3 — Grid Resolution Scaling (h-refinement study).
%
%  Problem:
%    Same heterogeneous Navier-Cauchy setup as exp_baseline_comparison:
%    aluminium background (mu=26, lambda=51) with centred steel inclusion
%    (mu=77, lambda=115; ~3× contrast), uniform downward body force fy=-1.
%    Grid is refined uniformly: nx = ny in {16, 32, 64, 128}.
%
%  Goal:
%    Measure whether each preconditioner achieves mesh-independent iteration
%    counts and O(N) total solve time as N = 2*nx*ny grows.
%
%  Per-grid outputs  (auto-saved to ../figures/):
%    navier_cauchy_NxN_*_setup.png
%    navier_cauchy_NxN_*_hierarchy_complexity.png
%    navier_cauchy_NxN_*_convergence.png
%
%  Scaling summary outputs  (auto-saved to ../figures/):
%    scaling_iterations.png   — GMRES iterations vs N  (log-log)
%    scaling_total_time.png   — total wall time vs N   (log-log)
%
%  Console output:
%    Per-grid summary tables (same format as exp_baseline_comparison) plus
%    a cross-grid scaling table showing iterations and total time for every
%    method × grid size combination.
% =========================================================================
clear; clc; close all;

% -------------------------------------------------------------------------
% 1. Shared configuration
% -------------------------------------------------------------------------
grid_sizes = [16, 32, 64, 128];
% grid_sizes = [16, 32, 64, 128, 256];  % uncomment for full sweep (slow for plain/GS)

% Material: identical to exp_baseline_comparison
mu_bg = 26;   lambda_bg  = 51;
mu_inc = 77;  lambda_inc = 115;
r_inc  = 0.2; cx = 0.5;  cy = 0.5;
contrast = mu_inc / mu_bg;

% Solver options: identical to exp_baseline_comparison
gmres_opts.tol      = 1e-8;
gmres_opts.max_iter = 300;
gmres_opts.restart  = 50;

amg_opts.theta            = 0.25;
amg_opts.max_levels       = 8;
amg_opts.coarse_threshold = 40;
amg_opts.nu1              = 1;
amg_opts.nu2              = 1;

pc_opts.methods = {'plain', 'ilu', 'gauss-seidel', 'block-jacobi', 'geomg', 'block-amg'};

% Display labels and plot styles  (must match pc_opts.methods order)
meth_labels = {'Plain GMRES', 'ILU(0)', 'Gauss-Seidel', 'Block Jacobi', 'Geom MG', 'Block AMG'};
meth_clrs   = {[0.00, 0.00, 0.00], ...  % plain        — black
               [0.85, 0.10, 0.10], ...  % ilu          — red
               [0.00, 0.50, 0.00], ...  % gauss-seidel — dark green
               [0.55, 0.00, 0.75], ...  % block-jacobi — purple
               [0.85, 0.45, 0.10], ...  % geomg        — orange
               [0.00, 0.65, 0.60]};     % block-amg    — teal
meth_marks  = {'o', 's', 'd', '^', 'v', 'p'};
meth_lspecs = {'--', '-', '-', '-', '-', '-'};

% Figures output directory
script_dir = fileparts(mfilename('fullpath'));
fig_dir    = fullfile(script_dir, '..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% -------------------------------------------------------------------------
% 2. Main loop over grid sizes
% -------------------------------------------------------------------------
n_grids   = length(grid_sizes);
n_methods = length(pc_opts.methods);

% Preallocate scaling data arrays
all_N          = zeros(1, n_grids);
all_nnz        = zeros(1, n_grids);
all_iters      = zeros(n_methods, n_grids);
all_flags      = zeros(n_methods, n_grids);
all_setup_time = zeros(n_methods, n_grids);
all_solve_time = zeros(n_methods, n_grids);
all_resid      = zeros(n_methods, n_grids);

fprintf('=================================================================\n');
fprintf('  Experiment 3: Grid Resolution Scaling\n');
fprintf('  Contrast: %.0fx   Methods: %s\n', contrast, strjoin(pc_opts.methods, ', '));
fprintf('=================================================================\n\n');

for g = 1:n_grids

    % Close figures from the previous grid size (they are already saved)
    if g > 1
        close all;
    end

    nx = grid_sizes(g);
    ny = nx;
    h  = 1 / (nx + 1);
    N  = 2 * nx * ny;   % total DOFs

    fprintf('------------------------------------------------------------\n');
    fprintf('  Grid %d×%d   (N = %d DOFs)\n', nx, ny, N);
    fprintf('------------------------------------------------------------\n');

    % Build material fields
    x_vec = linspace(h, 1 - h, nx);
    y_vec = linspace(h, 1 - h, ny);
    [X, Y] = meshgrid(x_vec, y_vec);

    in_inc     = ((X - cx).^2 + (Y - cy).^2) <= r_inc^2;
    mu_arr     = mu_bg     * ones(ny, nx);
    lambda_arr = lambda_bg * ones(ny, nx);
    mu_arr    (in_inc) = mu_inc;
    lambda_arr(in_inc) = lambda_inc;

    fx = zeros(ny, nx);
    fy = -ones(ny, nx);   % uniform downward body force (deterministic across grid sizes)

    % Assemble system
    asm_opts.dof_ordering = 'node';
    [K, b] = build_navier_cauchy_heterogeneous(nx, ny, h, lambda_arr, mu_arr, fx, fy, asm_opts);
    fprintf('  nnz(K) = %d\n\n', nnz(K));

    all_N  (g) = size(K, 1);
    all_nnz(g) = nnz(K);

    % Problem struct (identical layout to exp_baseline_comparison)
    problem.A            = K;
    problem.b            = b;
    problem.name         = sprintf('Navier-Cauchy %dx%d contrast %.0fx', nx, ny, contrast);
    problem.dof_per_node = 2;

    problem.grid_meta.nx           = nx;
    problem.grid_meta.ny           = ny;
    problem.grid_meta.hx           = h;
    problem.grid_meta.hy           = h;
    problem.grid_meta.dof_per_node = 2;

    % Run experiment — produces and saves all three per-grid figures
    results = run_preconditioner_experiment(problem, gmres_opts, amg_opts, pc_opts);

    % Per-grid summary table (same format as exp_baseline_comparison)
    fprintf('\n');
    fprintf('================================================================\n');
    fprintf('  SUMMARY  (%d×%d grid, %d DOFs, contrast %.0f×)\n', nx, ny, size(K,1), contrast);
    fprintf('================================================================\n');
    fprintf('%-22s %8s %12s %10s %10s\n', 'Method', 'Iters', 'Rel.Resid', 'Setup(s)', 'Solve(s)');
    fprintf('%s\n', repmat('-', 1, 66));
    for m = 1:numel(results)
        r = results(m);
        if isnan(r.setup_time)
            st = '       N/A';
        else
            st = sprintf('%10.3f', r.setup_time);
        end
        conv_flag = '';
        if r.flag ~= 0, conv_flag = '*'; end
        fprintf('%-22s %7d%s %12.2e %10s %10.3f\n', ...
                r.method, r.iter, conv_flag, r.true_resid, st, r.solve_time);
    end
    fprintf('  (* = did not reach tolerance within max_iter)\n');
    fprintf('================================================================\n\n');

    % Collect into scaling arrays
    for m = 1:n_methods
        all_iters     (m, g) = results(m).iter;
        all_flags     (m, g) = results(m).flag;
        all_resid     (m, g) = results(m).true_resid;
        st = results(m).setup_time;
        if isnan(st), st = 0; end
        all_setup_time(m, g) = st;
        all_solve_time(m, g) = results(m).solve_time;
    end

end   % grid loop

% -------------------------------------------------------------------------
% 3. Cross-grid scaling summary table
% -------------------------------------------------------------------------
fprintf('\n');
fprintf('=======================================================================\n');
fprintf('  SCALING SUMMARY — Iterations  (* = did not converge)\n');
fprintf('  Contrast %.0f×,  tol = %.0e\n', contrast, gmres_opts.tol);
fprintf('=======================================================================\n');
header = sprintf('%-22s', 'Method');
for g = 1:n_grids
    header = [header, sprintf(' %8s', sprintf('%dx%d', grid_sizes(g), grid_sizes(g)))]; %#ok<AGROW>
end
fprintf('%s\n', header);
fprintf('%s\n', repmat('-', 1, 22 + 9*n_grids));
for m = 1:n_methods
    row = sprintf('%-22s', pc_opts.methods{m});
    for g = 1:n_grids
        flag_str = '';
        if all_flags(m, g) ~= 0, flag_str = '*'; end
        row = [row, sprintf(' %7d%s ', all_iters(m, g), flag_str)]; %#ok<AGROW>
    end
    fprintf('%s\n', row);
end
fprintf('=======================================================================\n\n');

fprintf('=======================================================================\n');
fprintf('  SCALING SUMMARY — Total Time / s  (setup + solve)\n');
fprintf('=======================================================================\n');
fprintf('%s\n', header);
fprintf('%s\n', repmat('-', 1, 22 + 9*n_grids));
for m = 1:n_methods
    row = sprintf('%-22s', pc_opts.methods{m});
    for g = 1:n_grids
        t_tot = all_setup_time(m, g) + all_solve_time(m, g);
        row = [row, sprintf(' %9.2f', t_tot)]; %#ok<AGROW>
    end
    fprintf('%s\n', row);
end
fprintf('=======================================================================\n\n');

% -------------------------------------------------------------------------
% 4. Scaling figures
% -------------------------------------------------------------------------

% --- Fig A: Iterations vs N (log-log) ------------------------------------
fig_iters = figure('Name', 'Scaling: Iterations vs N', 'Position', [100 100 700 480]);
set(gca, 'XScale', 'log', 'YScale', 'log');
hold on;

for m = 1:n_methods
    clr  = meth_clrs{m};
    lsp  = meth_lspecs{m};
    mk   = meth_marks{m};

    % Split into converged and non-converged points for distinct markers
    conv_mask = (all_flags(m, :) == 0);

    % Plot one continuous line for context
    loglog(all_N, all_iters(m, :), lsp, ...
        'Color', clr, 'LineWidth', 1.6, 'HandleVisibility', 'off');

    % Converged points: filled marker
    if any(conv_mask)
        loglog(all_N(conv_mask), all_iters(m, conv_mask), ...
            [lsp, mk], 'Color', clr, 'LineWidth', 1.6, ...
            'MarkerSize', 8, 'MarkerFaceColor', clr, ...
            'DisplayName', meth_labels{m});
    end

    % Non-converged points: open marker with warning ring
    if any(~conv_mask)
        loglog(all_N(~conv_mask), all_iters(m, ~conv_mask), ...
            mk, 'Color', clr, 'LineWidth', 1.6, ...
            'MarkerSize', 10, 'MarkerFaceColor', 'none', ...
            'HandleVisibility', 'off');
        % Annotate with asterisk
        for g = find(~conv_mask)
            text(all_N(g), all_iters(m, g) * 1.15, '*', ...
                'Color', clr, 'FontSize', 10, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');
        end
        % Add legend entry only if no converged points exist for this method
        if ~any(conv_mask)
            loglog(NaN, NaN, [lsp, mk], 'Color', clr, 'LineWidth', 1.6, ...
                'MarkerSize', 8, 'MarkerFaceColor', 'none', ...
                'DisplayName', [meth_labels{m}, ' (*)']);
        end
    end
end

% Reference lines for O(1) and O(log N) scaling
N_ref   = [all_N(1), all_N(end)];
itr_mid = median(all_iters(:));
loglog(N_ref, repmat(itr_mid * 0.7, 1, 2), 'k:', 'LineWidth', 1.0, ...
    'HandleVisibility', 'off');
text(N_ref(end) * 1.05, itr_mid * 0.7, 'O(1)', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);

xlabel('N  (total DOFs)', 'FontSize', 12);
ylabel('GMRES Iterations', 'FontSize', 12);
title(sprintf('Iteration Count vs Problem Size\n(contrast %.0f\\times, tol=%.0e,  * = max iter reached)', ...
              contrast, gmres_opts.tol), 'FontSize', 11);
legend('Location', 'northwest', 'FontSize', 10);
grid on;
xticks(all_N);
xticklabels(arrayfun(@(n) sprintf('%d', n), all_N, 'UniformOutput', false));

print(fig_iters, fullfile(fig_dir, 'scaling_iterations.png'), '-dpng', '-r150');
fprintf('  [saved] scaling_iterations.png\n');

% --- Fig B: Total time vs N (log-log) ------------------------------------
fig_time = figure('Name', 'Scaling: Total Time vs N', 'Position', [820 100 700 480]);
set(gca, 'XScale', 'log', 'YScale', 'log');
hold on;

for m = 1:n_methods
    clr = meth_clrs{m};
    lsp = meth_lspecs{m};
    mk  = meth_marks{m};
    t_tot = all_setup_time(m, :) + all_solve_time(m, :);

    conv_mask = (all_flags(m, :) == 0);

    loglog(all_N, t_tot, lsp, ...
        'Color', clr, 'LineWidth', 1.6, 'HandleVisibility', 'off');

    if any(conv_mask)
        loglog(all_N(conv_mask), t_tot(conv_mask), ...
            [lsp, mk], 'Color', clr, 'LineWidth', 1.6, ...
            'MarkerSize', 8, 'MarkerFaceColor', clr, ...
            'DisplayName', meth_labels{m});
    end
    if any(~conv_mask)
        loglog(all_N(~conv_mask), t_tot(~conv_mask), ...
            mk, 'Color', clr, 'LineWidth', 1.6, ...
            'MarkerSize', 10, 'MarkerFaceColor', 'none', ...
            'HandleVisibility', 'off');
        if ~any(conv_mask)
            loglog(NaN, NaN, [lsp, mk], 'Color', clr, 'LineWidth', 1.6, ...
                'MarkerSize', 8, 'MarkerFaceColor', 'none', ...
                'DisplayName', [meth_labels{m}, ' (*)']);
        end
    end
end

% Reference slope lines: O(N) and O(N^2)
t_ref  = all_setup_time(end, end) + all_solve_time(end, end);  % anchor at largest N
N2 = all_N(end);
slope1 = t_ref * (N_ref / N2);            % O(N)
slope2 = t_ref * (N_ref / N2).^1.5;      % O(N^1.5)
loglog(N_ref, slope1 * 0.5, 'k:',  'LineWidth', 1.0, 'HandleVisibility', 'off');
loglog(N_ref, slope2 * 0.3, 'k--', 'LineWidth', 1.0, 'HandleVisibility', 'off');
text(N2 * 1.05, slope1(end) * 0.5,   'O(N)',     'FontSize', 8, 'Color', [0.4 0.4 0.4]);
text(N2 * 1.05, slope2(end) * 0.3,   'O(N^{1.5})', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);

xlabel('N  (total DOFs)', 'FontSize', 12);
ylabel('Wall Time / s  (setup + solve)', 'FontSize', 12);
title(sprintf('Total Wall Time vs Problem Size\n(contrast %.0f\\times, * = max iter reached)', ...
              contrast), 'FontSize', 11);
legend('Location', 'northwest', 'FontSize', 10);
grid on;
xticks(all_N);
xticklabels(arrayfun(@(n) sprintf('%d', n), all_N, 'UniformOutput', false));

print(fig_time, fullfile(fig_dir, 'scaling_total_time.png'), '-dpng', '-r150');
fprintf('  [saved] scaling_total_time.png\n\n');

fprintf('Experiment 3 complete.\n');
fprintf('Per-grid figures:   ../figures/navier_cauchy_*_{setup,hierarchy_complexity,convergence}.png\n');
fprintf('Scaling figures:    ../figures/scaling_iterations.png\n');
fprintf('                    ../figures/scaling_total_time.png\n');
