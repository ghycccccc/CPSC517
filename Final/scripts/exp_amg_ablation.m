% =========================================================================
%  exp_amg_ablation.m
%
%  Experiment 5 — AMG Design Ablation Study.
%
%  Isolates the contribution of two independent AMG design decisions on the
%  2D heterogeneous Navier-Cauchy problem:
%
%    Axis 1 — Prolongation strategy  (M1 vs M2 vs M3/M4):
%      Scalar AMG:     per-DOF CF split, classical RS scalar interpolation.
%      Block AMG RS:   node-level CF split, RS block weights (2×2 ω_ij matrices),
%                      no SA smoothing.  Satisfies AMG interpolation property
%                      directly via local equilibrium equation in block form.
%      Block AMG prop: node-level CF split, proportional Frobenius-norm weights
%                      + SA-style Jacobi prolongation smoothing.
%      M1→M2 tests node-level coarsening + RS block weights vs scalar AMG.
%      M2→M3 tests RS block weights vs proportional+SA (same block hierarchy).
%
%    Axis 2 — Smoother  (M3 vs M4, identical proportional+SA hierarchy):
%      Scalar GS: standard tril/triu sweep; treats all DOFs independently.
%      Block GS:  2×2 block tril/triu sweep; updates (u_i, v_i) jointly.
%
%  Methods:
%    M1 — Scalar AMG  + scalar GS     (amg_setup        + amg_preconditioner)
%    M2 — Block AMG RS + block GS     (block_amg_setup_rs + vcycle 'block')
%    M3 — Block AMG prop+SA + scalar GS  (block_amg_setup  + vcycle 'scalar')
%    M4 — Block AMG prop+SA + block  GS  (block_amg_setup  + vcycle 'block')
%
%  M3 and M4 share an identical hierarchy; only the smoother differs.
%  M2 uses a separate RS hierarchy (block_amg_setup_rs).
%
%  Problem:
%    nx×ny uniform grid (set below; default 64×64), h = 1/(nx+1).
%    Background (aluminium): mu=26, lambda=51.
%    Stiff circular inclusion (~3x): mu=77, lambda=115.
%    Uniform downward body force fy=-1.  Zero Dirichlet BCs.
%
%  Outputs (auto-saved to ../figures/):
%    amg_ablation_residuals.png    — loglog GMRES residual histories
%    amg_ablation_iterations.png  — iteration count bar chart
%    amg_ablation_timing.png      — stacked setup/solve time bar chart
%    amg_ablation_complexity.png  — nnz and DOF per hierarchy level
% =========================================================================
clear; clc; close all;

% -------------------------------------------------------------------------
% 0. Output directory
% -------------------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
fig_dir    = fullfile(script_dir, '..', 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

% -------------------------------------------------------------------------
% 1. Problem: 32x32 heterogeneous Navier-Cauchy
% -------------------------------------------------------------------------
nx = 64;  ny = 64;
h  = 1 / (nx + 1);

mu_bg = 26;  lambda_bg = 51;
mu_inc = 77; lambda_inc = 115;   % ~3x contrast
r_inc = 0.2; cx = 0.5; cy = 0.5;

x_vec = linspace(h, 1-h, nx);
y_vec = linspace(h, 1-h, ny);
[X, Y] = meshgrid(x_vec, y_vec);
in_inc = ((X - cx).^2 + (Y - cy).^2) <= r_inc^2;

mu_arr     = mu_bg     * ones(ny, nx);
lambda_arr = lambda_bg * ones(ny, nx);
mu_arr(in_inc)     = mu_inc;
lambda_arr(in_inc) = lambda_inc;

asm_opts.dof_ordering = 'node';
[K, b] = build_navier_cauchy_heterogeneous( ...
    nx, ny, h, lambda_arr, mu_arr, zeros(ny,nx), -ones(ny,nx), asm_opts);

n_dofs = size(K, 1);
normb  = norm(b);
if normb == 0, normb = 1; end
contrast = mu_inc / mu_bg;

fprintf('Problem: %d×%d Navier-Cauchy,  contrast ~%.0f×\n', nx, ny, contrast);
fprintf('DOFs: %d,  nnz(K): %d\n\n', n_dofs, nnz(K));

% -------------------------------------------------------------------------
% 2. Shared AMG / GMRES options
% -------------------------------------------------------------------------
theta            = 0.25;
max_levels       = 8;
coarse_threshold = 40;
nu1 = 1;  nu2 = 1;

gmres_tol      = 1e-6;
gmres_max_iter = 10;
gmres_restart  = 10;

% -------------------------------------------------------------------------
% 3. Build hierarchies
%    Scalar AMG      -> M1
%    Block AMG RS    -> M2  (RS block weights, no SA smoothing)
%    Block AMG prop  -> M3/M4 (proportional weights + SA; same hierarchy, smoother differs)
% -------------------------------------------------------------------------
fprintf('Building scalar AMG hierarchy (M1)...\n');
t0 = tic;
h_scalar = amg_setup(K, theta, max_levels, coarse_threshold);
t_setup_scalar = toc(t0);
fprintf('  Levels: %d,  OC: %.2f×,  GC: %.2f×,  time: %.3f s\n\n', ...
    length(h_scalar), hier_oc(h_scalar), hier_gc(h_scalar), t_setup_scalar);

fprintf('Building block AMG RS hierarchy (M2)...\n');
t0 = tic;
h_block_rs = block_amg_setup_rs(K, theta, max_levels, coarse_threshold);
t_setup_rs = toc(t0);
fprintf('  Levels: %d,  OC: %.2f×,  GC: %.2f×,  time: %.3f s\n\n', ...
    length(h_block_rs), hier_oc(h_block_rs), hier_gc(h_block_rs), t_setup_rs);

fprintf('Building block AMG prop+SA hierarchy (shared M3/M4)...\n');
t0 = tic;
h_block = block_amg_setup(K, theta, max_levels, coarse_threshold);
t_setup_block = toc(t0);
fprintf('  Levels: %d,  OC: %.2f×,  GC: %.2f×,  time: %.3f s\n\n', ...
    length(h_block), hier_oc(h_block), hier_gc(h_block), t_setup_block);

% -------------------------------------------------------------------------
% 4. Method configuration
%    Each entry: label, short label, preconditioner handle, setup time, style
% -------------------------------------------------------------------------
cfg(1).label      = 'M1: Scalar AMG + scalar GS';
cfg(1).short      = 'sAMG(sGS)';
cfg(1).apply      = @(r) amg_preconditioner(h_scalar, r, nu1, nu2);
cfg(1).setup_time = t_setup_scalar;
cfg(1).clr        = [0.00, 0.25, 0.90];   % blue
cfg(1).lspec      = '-';
cfg(1).lw         = 2.2;

cfg(2).label      = 'M2: Block AMG RS + block GS';
cfg(2).short      = 'bAMG-RS(bGS)';
cfg(2).apply      = @(r) block_amg_vcycle(h_block_rs, 1, r, zeros(n_dofs,1), nu1, nu2, 'block');
cfg(2).setup_time = t_setup_rs;
cfg(2).clr        = [0.75, 0.10, 0.15];   % crimson
cfg(2).lspec      = '-';
cfg(2).lw         = 2.2;

cfg(3).label      = 'M3: Block AMG prop+SA + scalar GS';
cfg(3).short      = 'bAMG(sGS)';
cfg(3).apply      = @(r) block_amg_vcycle(h_block, 1, r, zeros(n_dofs,1), nu1, nu2, 'scalar');
cfg(3).setup_time = t_setup_block;
cfg(3).clr        = [0.85, 0.45, 0.10];   % orange
cfg(3).lspec      = '-';
cfg(3).lw         = 2.2;

cfg(4).label      = 'M4: Block AMG prop+SA + block GS';
cfg(4).short      = 'bAMG(bGS)';
cfg(4).apply      = @(r) block_amg_vcycle(h_block, 1, r, zeros(n_dofs,1), nu1, nu2, 'block');
cfg(4).setup_time = t_setup_block;
cfg(4).clr        = [0.00, 0.65, 0.60];   % teal
cfg(4).lspec      = '--';
cfg(4).lw         = 1.8;



n_meth = numel(cfg);

% -------------------------------------------------------------------------
% 5. Run right-preconditioned GMRES for each method
%    A * M^{-1} y = b,  x = M^{-1} y
% -------------------------------------------------------------------------
fprintf('%-35s %8s %12s %10s %10s\n', 'Method', 'Iters', 'TrueResid', 'Setup(s)', 'Solve(s)');
fprintf('%s\n', repmat('-', 1, 70));

for m = 1:n_meth
    Afun = @(y) K * cfg(m).apply(y);

    t0 = tic;
    [y, flag, ~, iter_out, resvec] = gmres( ...
        Afun, b, gmres_restart, gmres_tol, gmres_max_iter);
    cfg(m).solve_time = toc(t0);

    cfg(m).x         = cfg(m).apply(y);
    cfg(m).iter      = flatten_iter(iter_out, gmres_restart);
    cfg(m).flag      = flag;
    cfg(m).resvec    = resvec;
    cfg(m).true_resid = norm(b - K * cfg(m).x) / normb;

    star = '';
    if flag ~= 0, star = '*'; end
    fprintf('%-35s %7d%s %12.2e %10.3f %10.3f\n', ...
        cfg(m).label, cfg(m).iter, star, cfg(m).true_resid, ...
        cfg(m).setup_time, cfg(m).solve_time);
end
fprintf('  (* = did not reach tolerance within max_iter)\n\n');

% -------------------------------------------------------------------------
% 6. Figure 1 — Residual histories (loglog)
% -------------------------------------------------------------------------
fig1 = figure('Name', 'AMG Ablation: Residual Histories', ...
              'Position', [50 50 720 480]);
set(gca, 'XScale', 'log', 'YScale', 'log');
hold on;

for m = 1:n_meth
    rv = cfg(m).resvec;
    if isempty(rv), continue; end
    rel_rv = rv / rv(1);
    iax = 0:length(rel_rv)-1;
    iax(iax == 0) = 1;
    loglog(iax, rel_rv, cfg(m).lspec, ...
        'Color', cfg(m).clr, 'LineWidth', cfg(m).lw, ...
        'DisplayName', cfg(m).label);
end

yline(gmres_tol, 'k:', 'LineWidth', 1.2, ...
    'DisplayName', sprintf('tol = %.0e', gmres_tol));
grid on;
xlabel('GMRES Iteration', 'FontSize', 12);
ylabel('Normalised Residual  |r_k| / |r_0|', 'FontSize', 12);
title(sprintf('AMG Ablation — Residual Histories\n(%d×%d, contrast ~%.0f×,  \\nu_1=\\nu_2=%d)', ...
              nx, ny, contrast, nu1), 'FontSize', 11);
legend('Location', 'southeast', 'FontSize', 10);
save_fig(fig1, fig_dir, 'amg_ablation_residuals');

% -------------------------------------------------------------------------
% 7. Figure 2 — Iteration count bar chart
% -------------------------------------------------------------------------
fig2 = figure('Name', 'AMG Ablation: Iteration Count', ...
              'Position', [780 50 520 420]);
hold on;

iters = [cfg.iter];
flags = [cfg.flag];

for m = 1:n_meth
    bar(m, iters(m), 0.55, ...
        'FaceColor', cfg(m).clr, 'EdgeColor', 'k', 'LineWidth', 1.0);
end

% Annotate with exact count; asterisk if not converged
max_iter_total = gmres_max_iter * gmres_restart;   % total-iteration cap
y_ceil = max(max(iters) * 1.20, 1);               % dynamic y headroom
for m = 1:n_meth
    lbl = sprintf('%d', iters(m));
    if flags(m) ~= 0, lbl = [lbl, '*']; end  %#ok<AGROW>
    % Place text just above bar; clamp so it stays inside axes
    text(m, min(iters(m) + y_ceil * 0.03, y_ceil * 0.96), lbl, ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
end

% Horizontal line at the total-iteration ceiling (restart × max_outer_cycles)
yline(max_iter_total, 'k:', 'LineWidth', 1.2);
set(gca, 'XTick', 1:n_meth, 'XTickLabel', {cfg.short}, 'FontSize', 10);
ylabel('GMRES Iterations to Convergence', 'FontSize', 12);
title(sprintf('AMG Ablation — Iteration Count\n(%d×%d, contrast ~%.0f×,  * = not converged)', ...
              nx, ny, contrast), 'FontSize', 11);
ylim([0, y_ceil]);
grid on;
save_fig(fig2, fig_dir, 'amg_ablation_iterations');

% -------------------------------------------------------------------------
% 8. Figure 3 — Stacked setup / solve time
% -------------------------------------------------------------------------
fig3 = figure('Name', 'AMG Ablation: Timing', ...
              'Position', [50 560 520 420]);

setup_t = [cfg.setup_time];
solve_t = [cfg.solve_time];

% bar() with [n x 2] matrix produces n stacked groups
bh = bar([setup_t; solve_t]', 'stacked', 'BarWidth', 0.55);
bh(1).FaceColor = [0.65, 0.65, 0.65];   % setup: grey
bh(2).FaceColor = [0.20, 0.55, 0.85];   % solve: sky blue
set(gca, 'YScale', 'log');

% Total time label above each bar (multiplicative offset for log scale)
for m = 1:n_meth
    t_tot = setup_t(m) + solve_t(m);
    text(m, t_tot * 1.18, sprintf('%.2f s', t_tot), ...
        'HorizontalAlignment', 'center', 'FontSize', 10);
end

set(gca, 'XTick', 1:n_meth, 'XTickLabel', {cfg.short}, 'FontSize', 10);
ylabel('Time (s)', 'FontSize', 12);
title(sprintf('AMG Ablation — Setup + Solve Time\n(%d×%d, contrast ~%.0f×;  M3/M4 share setup)', ...
              nx, ny, contrast), 'FontSize', 11);
legend(bh, {'Setup', 'Solve'}, 'Location', 'northwest', 'FontSize', 10);
grid on;
save_fig(fig3, fig_dir, 'amg_ablation_timing');

% -------------------------------------------------------------------------
% 9. Figure 4 — Hierarchy complexity: nnz and DOF per level
%    Two lines: scalar AMG (M1) vs block AMG (shared M2/M3).
% -------------------------------------------------------------------------
fig4 = figure('Name', 'AMG Ablation: Hierarchy Complexity', ...
              'Position', [780 560 880 400]);

clr_s  = [0.00, 0.25, 0.90];   % blue    — scalar AMG  (M1)
clr_rs = [0.75, 0.10, 0.15];   % crimson — block AMG RS (M2)
clr_b  = [0.00, 0.65, 0.60];   % teal    — block AMG prop+SA (M3/M4)

L_s  = length(h_scalar);
L_rs = length(h_block_rs);
L_b  = length(h_block);

s_nnz   = arrayfun(@(l) nnz(h_scalar{l}.A),    1:L_s);
s_dofs  = arrayfun(@(l) size(h_scalar{l}.A,1),  1:L_s);
rs_nnz  = arrayfun(@(l) nnz(h_block_rs{l}.A),  1:L_rs);
rs_dofs = arrayfun(@(l) size(h_block_rs{l}.A,1),1:L_rs);
b_nnz   = arrayfun(@(l) nnz(h_block{l}.A),     1:L_b);
b_dofs  = arrayfun(@(l) size(h_block{l}.A,1),   1:L_b);

oc_s  = sum(s_nnz)  / s_nnz(1);    gc_s  = sum(s_dofs)  / s_dofs(1);
oc_rs = sum(rs_nnz) / rs_nnz(1);   gc_rs = sum(rs_dofs) / rs_dofs(1);
oc_b  = sum(b_nnz)  / b_nnz(1);    gc_b  = sum(b_dofs)  / b_dofs(1);
L_max = max([L_s, L_rs, L_b]);

% Left panel: nnz per level
subplot(1, 2, 1);
hold on;
semilogy(1:L_s,  s_nnz,  '-o', 'Color', clr_s,  'LineWidth', 1.8, ...
    'MarkerSize', 7, 'MarkerFaceColor', clr_s, ...
    'DisplayName', sprintf('M1 Scalar AMG   (OC=%.2f×)', oc_s));
semilogy(1:L_rs, rs_nnz, '-^', 'Color', clr_rs, 'LineWidth', 1.8, ...
    'MarkerSize', 7, 'MarkerFaceColor', clr_rs, ...
    'DisplayName', sprintf('M2 Block AMG RS  (OC=%.2f×)', oc_rs));
semilogy(1:L_b,  b_nnz,  '-s', 'Color', clr_b,  'LineWidth', 1.8, ...
    'MarkerSize', 7, 'MarkerFaceColor', clr_b, ...
    'DisplayName', sprintf('M3/M4 Block AMG  (OC=%.2f×)', oc_b));
annotate_pts(1:L_s,  s_nnz,  clr_s,  'above');
annotate_pts(1:L_rs, rs_nnz, clr_rs, 'below');
annotate_pts(1:L_b,  b_nnz,  clr_b,  'above');
set(gca, 'YScale', 'log');
xlabel('Level');  ylabel('nnz(A_l)');
title('nnz per hierarchy level', 'FontSize', 10);
xticks(1:L_max);  xlim([0.5, L_max + 0.5]);
legend('Location', 'southwest', 'FontSize', 9);
grid on;

% Right panel: DOF count per level
subplot(1, 2, 2);
hold on;
semilogy(1:L_s,  s_dofs,  '-o', 'Color', clr_s,  'LineWidth', 1.8, ...
    'MarkerSize', 7, 'MarkerFaceColor', clr_s, ...
    'DisplayName', sprintf('M1 Scalar AMG   (GC=%.2f×)', gc_s));
semilogy(1:L_rs, rs_dofs, '-^', 'Color', clr_rs, 'LineWidth', 1.8, ...
    'MarkerSize', 7, 'MarkerFaceColor', clr_rs, ...
    'DisplayName', sprintf('M2 Block AMG RS  (GC=%.2f×)', gc_rs));
semilogy(1:L_b,  b_dofs,  '-s', 'Color', clr_b,  'LineWidth', 1.8, ...
    'MarkerSize', 7, 'MarkerFaceColor', clr_b, ...
    'DisplayName', sprintf('M3/M4 Block AMG  (GC=%.2f×)', gc_b));
annotate_pts(1:L_s,  s_dofs,  clr_s,  'above');
annotate_pts(1:L_rs, rs_dofs, clr_rs, 'below');
annotate_pts(1:L_b,  b_dofs,  clr_b,  'above');
set(gca, 'YScale', 'log');
xlabel('Level');  ylabel('DOFs at level l');
title('DOF count per hierarchy level', 'FontSize', 10);
xticks(1:L_max);  xlim([0.5, L_max + 0.5]);
legend('Location', 'southwest', 'FontSize', 9);
grid on;

sgtitle(sprintf('AMG Hierarchy Complexity  (%d×%d, contrast ~%.0f×)', ...
                nx, ny, contrast), 'FontSize', 12, 'FontWeight', 'bold');
save_fig(fig4, fig_dir, 'amg_ablation_complexity');

fprintf('\nExperiment 5 complete.\n');
fprintf('Figures saved to: %s\n', fig_dir);


% =========================================================================
%  Local helpers
% =========================================================================

function c = hier_oc(h)
    c = sum(cellfun(@(lv) nnz(lv.A), h)) / nnz(h{1}.A);
end

function c = hier_gc(h)
    c = sum(cellfun(@(lv) size(lv.A,1), h)) / size(h{1}.A,1);
end

function iter = flatten_iter(iter_out, restart)
    if numel(iter_out) == 2
        outer = iter_out(1);  inner = iter_out(2);
        if inner == 0
            iter = (outer - 1) * restart;
        else
            iter = (outer - 1) * restart + inner;
        end
    else
        iter = iter_out;
    end
end

function annotate_pts(xs, ys, clr, side)
    if strcmpi(side, 'above'), va = 'bottom'; yfac = 1.30;
    else,                      va = 'top';    yfac = 0.75;
    end
    for k = 1:numel(xs)
        v = ys(k);
        if v <= 0, continue; end
        if     v >= 1e6, lbl = sprintf('%.1fM', v/1e6);
        elseif v >= 1e3, lbl = sprintf('%.1fk', v/1e3);
        else,            lbl = sprintf('%d', round(v));
        end
        text(xs(k), v * yfac, lbl, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', va, ...
            'FontSize', 7.5, 'Color', clr * 0.75);
    end
end

function save_fig(fig, fig_dir, name)
    path = fullfile(fig_dir, [name, '.png']);
    print(fig, path, '-dpng', '-r150');
    fprintf('[saved] %s.png\n', name);
end
