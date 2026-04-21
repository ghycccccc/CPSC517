% =========================================================================
%  exp_contrast_sweep.m
%
%  Experiment 4 — Material Contrast Sweep.
%
%  Problem:
%    Fixed 64×64 Navier-Cauchy grid with a centred circular inclusion.
%    Background: mu_bg=26, lambda_bg=51 (aluminium-like).
%    Inclusion:  mu_inc = contrast * mu_bg,
%                lambda_inc = contrast * lambda_bg.
%    Scaling both Lame parameters by the same factor preserves Poisson's
%    ratio (nu = lambda/(2*(lambda+mu))) across all contrast values, so the
%    only physical change is the stiffness ratio between the two materials.
%    Contrast = 1 recovers the homogeneous (constant-coefficient) case.
%
%  Contrast values:  {1, 2, 5, 10, 50, 100}
%
%  Per-contrast outputs  (auto-saved to ../figures/):
%    navier_cauchy_*contrast*_setup.png
%    navier_cauchy_*contrast*_hierarchy_complexity.png
%    navier_cauchy_*contrast*_convergence.png
%
%  Sweep summary outputs  (auto-saved to ../figures/):
%    contrast_sweep_iterations.png   — iterations vs contrast  (log-x)
%    contrast_sweep_residuals.png    — final true residual vs contrast (log-x log-y)
%    contrast_sweep_bamg_histories.png — block-AMG convergence curves, all contrasts
%
%  Console output:
%    Per-contrast summary tables + two cross-contrast tables
%    (iterations and total time).
% =========================================================================
clear; clc; close all;

% -------------------------------------------------------------------------
% 1. Shared configuration
% -------------------------------------------------------------------------
nx = 64;  ny = 64;

contrast_values = [1, 2, 5, 10, 20];
%contrast_values = [1, 2, 5];

% Background material (fixed throughout)
mu_bg     = 26;
lambda_bg = 51;

% Geometry
r_inc = 0.2;  cx = 0.5;  cy = 0.5;

% Solver options (identical to exp_baseline_comparison)
gmres_opts.tol      = 1e-6;
gmres_opts.max_iter = 300;
gmres_opts.restart  = 50;

amg_opts.theta            = 0.25;
amg_opts.max_levels       = 8;
amg_opts.coarse_threshold = 40;
amg_opts.nu1              = 1;
amg_opts.nu2              = 1;

pc_opts.methods = {'plain', 'ilu', 'gauss-seidel', 'block-jacobi', 'geomg', 'block-amg'};

% Display labels and plot styles (must match pc_opts.methods order)
meth_labels = {'Plain GMRES', 'ILU(0)', 'Gauss-Seidel', 'Block Jacobi', 'Geom MG', 'Block AMG'};
meth_clrs   = {[0.00, 0.00, 0.00], ...   % plain
               [0.85, 0.10, 0.10], ...   % ilu
               [0.00, 0.50, 0.00], ...   % gauss-seidel
               [0.55, 0.00, 0.75], ...   % block-jacobi
               [0.85, 0.45, 0.10], ...   % geomg
               [0.00, 0.65, 0.60]};      % block-amg
meth_marks  = {'o', 's', 'd', '^', 'v', 'p'};
meth_lspecs = {'--', '-', '-', '-', '-', '-'};

% Figure output directory
script_dir = fileparts(mfilename('fullpath'));
fig_dir    = fullfile(script_dir, '..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% -------------------------------------------------------------------------
% 2. Precompute geometry mask (same for all contrasts)
% -------------------------------------------------------------------------
h     = 1 / (nx + 1);
x_vec = linspace(h, 1 - h, nx);
y_vec = linspace(h, 1 - h, ny);
[X, Y] = meshgrid(x_vec, y_vec);
in_inc = ((X - cx).^2 + (Y - cy).^2) <= r_inc^2;

fx = zeros(ny, nx);
fy = -ones(ny, nx);

% -------------------------------------------------------------------------
% 3. Main loop over contrast values
% -------------------------------------------------------------------------
n_contrasts = length(contrast_values);
n_methods   = length(pc_opts.methods);

% Preallocate scaling data
all_iters      = zeros(n_methods, n_contrasts);
all_flags      = zeros(n_methods, n_contrasts);
all_resid      = zeros(n_methods, n_contrasts);
all_setup_time = zeros(n_methods, n_contrasts);
all_solve_time = zeros(n_methods, n_contrasts);

% Store block-AMG residual vectors for overlay plot
bamg_idx   = find(strcmpi(pc_opts.methods, 'block-amg'), 1);
bamg_rvecs = cell(1, n_contrasts);   % resvec per contrast

fprintf('=================================================================\n');
fprintf('  Experiment 4: Material Contrast Sweep\n');
fprintf('  Grid: %d×%d   Methods: %s\n', nx, ny, strjoin(pc_opts.methods, ', '));
fprintf('=================================================================\n\n');

for c = 1:n_contrasts

    if c > 1
        close all;   % close previous contrast's figures (already saved)
    end

    contrast = contrast_values(c);

    mu_inc     = contrast * mu_bg;
    lambda_inc = contrast * lambda_bg;

    fprintf('------------------------------------------------------------\n');
    fprintf('  Contrast %.0f×   (mu: %g→%g,  lambda: %g→%g)\n', ...
            contrast, mu_bg, mu_inc, lambda_bg, lambda_inc);
    fprintf('------------------------------------------------------------\n');

    % Build material arrays
    mu_arr     = mu_bg     * ones(ny, nx);
    lambda_arr = lambda_bg * ones(ny, nx);
    mu_arr    (in_inc) = mu_inc;
    lambda_arr(in_inc) = lambda_inc;

    % Assemble
    asm_opts.dof_ordering = 'node';
    [K, b] = build_navier_cauchy_heterogeneous(nx, ny, h, lambda_arr, mu_arr, fx, fy, asm_opts);
    fprintf('  nnz(K) = %d\n\n', nnz(K));

    % Problem struct
    problem.A            = K;
    problem.b            = b;
    problem.name         = sprintf('Navier-Cauchy %dx%d contrast %.0fx', nx, ny, contrast);
    problem.dof_per_node = 2;

    problem.grid_meta.nx           = nx;
    problem.grid_meta.ny           = ny;
    problem.grid_meta.hx           = h;
    problem.grid_meta.hy           = h;
    problem.grid_meta.dof_per_node = 2;

    % Run experiment — saves the three per-contrast figures automatically
    results = run_preconditioner_experiment(problem, gmres_opts, amg_opts, pc_opts);

    % Per-contrast summary table
    fprintf('\n');
    fprintf('================================================================\n');
    fprintf('  SUMMARY  (%d×%d, contrast %.0f×, %d DOFs)\n', nx, ny, contrast, size(K,1));
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

    % Collect into sweep arrays
    for m = 1:n_methods
        all_iters     (m, c) = results(m).iter;
        all_flags     (m, c) = results(m).flag;
        all_resid     (m, c) = results(m).true_resid;
        st = results(m).setup_time;
        if isnan(st), st = 0; end
        all_setup_time(m, c) = st;
        all_solve_time(m, c) = results(m).solve_time;
    end

    % Save block-AMG residual vector for overlay figure
    if ~isempty(bamg_idx) && bamg_idx <= numel(results)
        rv = results(bamg_idx).resvec;
        if ~isempty(rv)
            bamg_rvecs{c} = rv / rv(1);   % normalised
        end
    end

end   % contrast loop

% -------------------------------------------------------------------------
% 4. Cross-contrast summary tables
% -------------------------------------------------------------------------
cstr_hdr = arrayfun(@(c) sprintf('%5.0fx', c), contrast_values, 'UniformOutput', false);
col_w    = 7;

fprintf('\n');
fprintf('=======================================================================\n');
fprintf('  SWEEP SUMMARY — Iterations  (* = did not converge)\n');
fprintf('  Grid %d×%d,  tol = %.0e\n', nx, ny, gmres_opts.tol);
fprintf('=======================================================================\n');
header = sprintf('%-22s', 'Method');
for c = 1:n_contrasts
    header = [header, sprintf(' %*s', col_w, cstr_hdr{c})]; 
end
fprintf('%s\n', header);
fprintf('%s\n', repmat('-', 1, 22 + (col_w+1)*n_contrasts));
for m = 1:n_methods
    row = sprintf('%-22s', pc_opts.methods{m});
    for c = 1:n_contrasts
        flag_str = ' ';
        if all_flags(m, c) ~= 0, flag_str = '*'; end
        row = [row, sprintf(' %*d%s', col_w-1, all_iters(m, c), flag_str)]; 
    end
    fprintf('%s\n', row);
end
fprintf('=======================================================================\n\n');

fprintf('=======================================================================\n');
fprintf('  SWEEP SUMMARY — Total Time / s  (setup + solve)\n');
fprintf('=======================================================================\n');
fprintf('%s\n', header);
fprintf('%s\n', repmat('-', 1, 22 + (col_w+1)*n_contrasts));
for m = 1:n_methods
    row = sprintf('%-22s', pc_opts.methods{m});
    for c = 1:n_contrasts
        t_tot = all_setup_time(m, c) + all_solve_time(m, c);
        row = [row, sprintf(' %*.2f', col_w, t_tot)]; 
    end
    fprintf('%s\n', row);
end
fprintf('=======================================================================\n\n');

% -------------------------------------------------------------------------
% 5. Sweep figures
% -------------------------------------------------------------------------

% --- Fig A: Iterations vs contrast (log-x) --------------------------------
fig_iters = figure('Name', 'Sweep: Iterations vs Contrast', 'Position', [100 100 700 480]);
set(gca, 'XScale', 'log', 'YScale', 'log');
hold on;

for m = 1:n_methods
    clr  = meth_clrs{m};
    lsp  = meth_lspecs{m};
    mk   = meth_marks{m};

    conv_mask = (all_flags(m, :) == 0);

    % Full connecting line
    loglog(contrast_values, all_iters(m, :), lsp, ...
        'Color', clr, 'LineWidth', 1.6, 'HandleVisibility', 'off');

    % Converged: filled markers
    if any(conv_mask)
        loglog(contrast_values(conv_mask), all_iters(m, conv_mask), ...
            [lsp, mk], 'Color', clr, 'LineWidth', 1.6, ...
            'MarkerSize', 8, 'MarkerFaceColor', clr, ...
            'DisplayName', meth_labels{m});
    end

    % Non-converged: open markers + asterisk
    if any(~conv_mask)
        loglog(contrast_values(~conv_mask), all_iters(m, ~conv_mask), ...
            mk, 'Color', clr, 'LineWidth', 1.6, ...
            'MarkerSize', 10, 'MarkerFaceColor', 'none', ...
            'HandleVisibility', 'off');
        for ci = find(~conv_mask)
            text(contrast_values(ci), all_iters(m, ci) * 1.08, '*', ...
                'Color', clr, 'FontSize', 10, 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'center');
        end
        if ~any(conv_mask)
            loglog(NaN, NaN, [lsp, mk], 'Color', clr, 'LineWidth', 1.6, ...
                'MarkerSize', 8, 'MarkerFaceColor', 'none', ...
                'DisplayName', [meth_labels{m}, ' (*)']);
        end
    end
end

% Horizontal reference: ideal mesh-independent preconditioner
yline(gmres_opts.max_iter, 'k:', 'LineWidth', 1.0, ...
    'DisplayName', sprintf('max\\_iter = %d', gmres_opts.max_iter));

xlabel('Contrast  \mu_{inc}/\mu_{bg}', 'FontSize', 12);
ylabel('GMRES Iterations', 'FontSize', 12);
title(sprintf('Iteration Count vs Material Contrast\n(%d\\times%d grid, tol=%.0e,  * = max iter reached)', ...
              nx, ny, gmres_opts.tol), 'FontSize', 11);
legend('Location', 'northwest', 'FontSize', 10);
xticks(contrast_values);
xticklabels(arrayfun(@(c) sprintf('%.0f\\times', c), contrast_values, 'UniformOutput', false));
grid on;

print(fig_iters, fullfile(fig_dir, 'contrast_sweep_iterations.png'), '-dpng', '-r150');
fprintf('  [saved] contrast_sweep_iterations.png\n');

% --- Fig B: True residual vs contrast (log-x, log-y) ---------------------
fig_resid = figure('Name', 'Sweep: Residual vs Contrast', 'Position', [820 100 700 480]);
set(gca, 'XScale', 'log', 'YScale', 'log');
hold on;

for m = 1:n_methods
    clr = meth_clrs{m};
    lsp = meth_lspecs{m};
    mk  = meth_marks{m};
    conv_mask = (all_flags(m, :) == 0);

    loglog(contrast_values, all_resid(m, :), lsp, ...
        'Color', clr, 'LineWidth', 1.6, 'HandleVisibility', 'off');
    if any(conv_mask)
        loglog(contrast_values(conv_mask), all_resid(m, conv_mask), ...
            [lsp, mk], 'Color', clr, 'LineWidth', 1.6, ...
            'MarkerSize', 8, 'MarkerFaceColor', clr, ...
            'DisplayName', meth_labels{m});
    end
    if any(~conv_mask)
        loglog(contrast_values(~conv_mask), all_resid(m, ~conv_mask), ...
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

yline(gmres_opts.tol, 'k:', 'LineWidth', 1.2, 'DisplayName', 'Tolerance');

xlabel('Contrast  \mu_{inc}/\mu_{bg}', 'FontSize', 12);
ylabel('True Relative Residual |b - Kx| / |b|', 'FontSize', 12);
title(sprintf('Final Residual vs Material Contrast\n(%d\\times%d grid,  * = max iter reached)', ...
              nx, ny), 'FontSize', 11);
legend('Location', 'northwest', 'FontSize', 10);
xticks(contrast_values);
xticklabels(arrayfun(@(c) sprintf('%.0f\\times', c), contrast_values, 'UniformOutput', false));
grid on;

print(fig_resid, fullfile(fig_dir, 'contrast_sweep_residuals.png'), '-dpng', '-r150');
fprintf('  [saved] contrast_sweep_residuals.png\n');

% --- Fig C: Block-AMG convergence histories, all contrasts overlaid ------
% Shows how the residual-reduction profile changes with contrast, which is
% more informative than the single-number iteration count alone.
if ~isempty(bamg_idx)
    fig_bamg = figure('Name', 'Block-AMG: Convergence vs Contrast', ...
                      'Position', [100 560 700 420]);
    set(gca, 'XScale', 'log', 'YScale', 'log');
    hold on;

    % Colormap: cool blue→warm red as contrast increases
    cmap = cool(n_contrasts);

    for c = 1:n_contrasts
        rv = bamg_rvecs{c};
        if isempty(rv), continue; end
        iax = 0:length(rv)-1;
        iax(iax == 0) = 1;   % avoid zero on potential log-x axis

        loglog(iax, rv, '-', ...
            'Color', cmap(c, :), 'LineWidth', 1.8, ...
            'DisplayName', sprintf('contrast %.0f\\times', contrast_values(c)));
    end

    yline(gmres_opts.tol, 'k:', 'LineWidth', 1.2, 'DisplayName', 'Tolerance');

    xlabel('GMRES Iteration', 'FontSize', 12);
    ylabel('Normalised Residual |r_k| / |r_0|', 'FontSize', 12);
    title(sprintf('Block AMG-GMRES: Convergence at Each Contrast Level\n(%d\\times%d grid)', ...
                  nx, ny), 'FontSize', 11);
    legend('Location', 'northeast', 'FontSize', 9);
    grid on;

    print(fig_bamg, fullfile(fig_dir, 'contrast_sweep_bamg_histories.png'), '-dpng', '-r150');
    fprintf('  [saved] contrast_sweep_bamg_histories.png\n');
end

fprintf('\nExperiment 4 complete.\n');
fprintf('Per-contrast figures:  ../figures/navier_cauchy_*contrast*_{setup,hierarchy_complexity,convergence}.png\n');
fprintf('Sweep summary:         ../figures/contrast_sweep_iterations.png\n');
fprintf('                       ../figures/contrast_sweep_residuals.png\n');
if ~isempty(bamg_idx)
    fprintf('Block AMG overlay:     ../figures/contrast_sweep_bamg_histories.png\n');
end
