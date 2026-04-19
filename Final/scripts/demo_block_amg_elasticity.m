% =========================================================================
%  demo_block_amg_elasticity.m
%
%  Benchmark: three solvers on the 2D heterogeneous Navier-Cauchy problem
%  with a stiff circular inclusion.
%
%    1. Block AMG-preconditioned GMRES  (new — node-level coarsening + BGS)
%    2. Scalar AMG-preconditioned GMRES (existing — for comparison)
%    3. Unpreconditioned GMRES          (baseline)
%
%  The block AMG preconditioner addresses the two main failure modes of
%  scalar AMG on vector problems:
%    - Scalar coarsening can assign u_i and v_i of the same node to
%      opposite C/F classes, breaking the intra-node coupling.
%    - Scalar Gauss-Seidel ignores the u-v coupling at each node;
%      block GS solves a 2x2 system per node instead.
%
%  Requires:
%    build_navier_cauchy_heterogeneous.m
%    block_amg_setup.m, block_amg_vcycle.m, solve_with_block_amg_gmres.m
%    amg_setup.m, amg_vcycle.m, amg_preconditioner.m, solve_with_amg_gmres.m
% =========================================================================
clear; clc; close all;

fprintf('=================================================================\n');
fprintf('  Block AMG vs Scalar AMG vs Unpreconditioned GMRES\n');
fprintf('  2D Heterogeneous Navier-Cauchy (Linear Elasticity)\n');
fprintf('=================================================================\n\n');

% -------------------------------------------------------------------------
% 1. PROBLEM SETUP
% -------------------------------------------------------------------------
nx = 128;
ny = 128;
h  = 1 / (nx + 1);

x_vec = linspace(h, 1-h, nx);
y_vec = linspace(h, 1-h, ny);
[X, Y] = meshgrid(x_vec, y_vec);

% Soft background (aluminum-like) with stiff circular inclusion (steel-like)
mu_bg  = 26;   lambda_bg  = 51;
%mu_bg  = 77;   lambda_bg  = 115;
mu_inc = 77;   lambda_inc = 115;
r_inc  = 0.2;  cx = 0.5;  cy = 0.5;

in_inc = ((X - cx).^2 + (Y - cy).^2) <= r_inc^2;

mu_arr     = mu_bg     * ones(ny, nx);
lambda_arr = lambda_bg * ones(ny, nx);
mu_arr    (in_inc) = mu_inc;
lambda_arr(in_inc) = lambda_inc;

fx = zeros(ny, nx);
fy = -ones(ny, nx);   % uniform downward body force

fprintf('Material setup:\n');
fprintf('  Background: mu=%g, lambda=%g\n', mu_bg, lambda_bg);
fprintf('  Inclusion:  mu=%g, lambda=%g  (contrast %.1fx)\n\n', ...
        mu_inc, lambda_inc, mu_inc / mu_bg);

% -------------------------------------------------------------------------
% 2. ASSEMBLE SYSTEM
% -------------------------------------------------------------------------
fprintf('Assembling Navier-Cauchy system...\n');
opts.dof_ordering = 'node';
[K, b] = build_navier_cauchy_heterogeneous(nx, ny, h, lambda_arr, mu_arr, fx, fy, opts);
n_dofs = size(K, 1);
fprintf('  Size: %d x %d   nnz: %d\n\n', n_dofs, n_dofs, nnz(K));

% -------------------------------------------------------------------------
% 3. SHARED SOLVER OPTIONS
% -------------------------------------------------------------------------
amg_opts.theta            = 0.25;
amg_opts.max_levels       = 8;
amg_opts.coarse_threshold = 40;
amg_opts.nu1              = 1;
amg_opts.nu2              = 1;

gmres_opts.tol      = 1e-9;
gmres_opts.max_iter = 300;
gmres_opts.restart  = 50;

% -------------------------------------------------------------------------
% 4. SOLVER 1: BLOCK AMG-PRECONDITIONED GMRES
% -------------------------------------------------------------------------
fprintf('--- Solver 1: Block AMG-GMRES ---\n');
[x_bamg, resvec_bamg, iter_bamg] = solve_with_block_amg_gmres(K, b, amg_opts, gmres_opts);
res_bamg = norm(b - K * x_bamg) / norm(b);
fprintf('  Verified relative residual: %.2e\n\n', res_bamg);

% -------------------------------------------------------------------------
% 5. SOLVER 2: SCALAR AMG-PRECONDITIONED GMRES
% -------------------------------------------------------------------------
fprintf('--- Solver 2: Scalar AMG-GMRES ---\n');
[x_amg, resvec_amg, iter_amg] = solve_with_amg_gmres(K, b, amg_opts, gmres_opts);
res_amg = norm(b - K * x_amg) / norm(b);
fprintf('  Verified relative residual: %.2e\n\n', res_amg);

% -------------------------------------------------------------------------
% 6. SOLVER 3: UNPRECONDITIONED GMRES (BASELINE)
% -------------------------------------------------------------------------
fprintf('--- Solver 3: Unpreconditioned GMRES ---\n');
t_plain = tic;
[x_plain, flag_plain, ~, iter_plain_out, resvec_plain] = gmres( ...
    K, b, gmres_opts.restart, gmres_opts.tol, gmres_opts.max_iter);
t_plain = toc(t_plain);

if numel(iter_plain_out) == 2
    outer = iter_plain_out(1);  inner = iter_plain_out(2);
    iter_plain = (outer - 1) * gmres_opts.restart + max(inner, 0);
else
    iter_plain = iter_plain_out;
end
res_plain = norm(b - K * x_plain) / norm(b);
fprintf('  Flag: %d   Iterations: %d   Time: %.3f s\n', ...
        flag_plain, iter_plain, t_plain);
fprintf('  Verified relative residual: %.2e\n\n', res_plain);

% -------------------------------------------------------------------------
% 7. SUMMARY TABLE
% -------------------------------------------------------------------------
fprintf('=================================================================\n');
fprintf('  SUMMARY  (%d x %d grid, %d DOFs)\n', nx, ny, n_dofs);
fprintf('=================================================================\n');
fprintf('%-28s %10s %12s\n', 'Method', 'Iterations', 'Rel.Resid.');
fprintf('%-28s %10d %12.2e\n', 'Block AMG-GMRES',   iter_bamg,  res_bamg);
fprintf('%-28s %10d %12.2e\n', 'Scalar AMG-GMRES',  iter_amg,   res_amg);
fprintf('%-28s %10d %12.2e\n', 'Unpreconditioned',  iter_plain, res_plain);
fprintf('=================================================================\n\n');

if iter_bamg > 0 && iter_amg > 0
    fprintf('Block AMG iteration reduction over scalar AMG: %.1fx\n\n', ...
            iter_amg / iter_bamg);
end

% -------------------------------------------------------------------------
% 8. CONVERGENCE PLOT
% -------------------------------------------------------------------------
figure('Name', 'Convergence Comparison', 'Position', [100 100 760 480]);

colors = {[0.1 0.5 0.9], [0.9 0.35 0.1], [0.4 0.4 0.4]};

semilogy(0:length(resvec_bamg)-1,  resvec_bamg  / resvec_bamg(1),  ...
         '-',  'Color', colors{1}, 'LineWidth', 2.0, 'DisplayName', 'Block AMG-GMRES');
hold on;
semilogy(0:length(resvec_amg)-1,   resvec_amg   / resvec_amg(1),   ...
         '--', 'Color', colors{2}, 'LineWidth', 1.8, 'DisplayName', 'Scalar AMG-GMRES');
semilogy(0:length(resvec_plain)-1, resvec_plain / resvec_plain(1), ...
         ':',  'Color', colors{3}, 'LineWidth', 1.5, 'DisplayName', 'Unpreconditioned');
yline(gmres_opts.tol, 'k-.', 'LineWidth', 1.2, 'DisplayName', 'Tolerance');

xlabel('GMRES Iteration', 'FontSize', 12);
ylabel('Relative Residual Norm', 'FontSize', 12);
title(sprintf('Solver Convergence — %dx%d Navier-Cauchy (contrast %.0fx)', ...
              nx, ny, mu_inc/mu_bg), 'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 11);
grid on;
xlim([0, max([length(resvec_bamg), length(resvec_amg), length(resvec_plain)]) - 1]);
ylim([1e-10, 2]);

% -------------------------------------------------------------------------
% 9. DISPLACEMENT FIELD  (block AMG solution)
% -------------------------------------------------------------------------
getNode = @(i,j) (j-1)*nx + i;

U     = zeros(ny, nx);
V_fld = zeros(ny, nx);
for j = 1:ny
    for i = 1:nx
        nd        = getNode(i, j);
        U(j,i)    = x_bamg(2*nd - 1);
        V_fld(j,i) = x_bamg(2*nd);
    end
end

figure('Name', 'Block AMG Solution', 'Position', [100 600 900 380]);

subplot(1,2,1);
imagesc(x_vec, y_vec, U);
colorbar;  axis equal tight;  set(gca, 'YDir', 'normal');
title('Horizontal Displacement u(x,y)  [Block AMG]', 'FontSize', 11);
xlabel('x');  ylabel('y');

subplot(1,2,2);
imagesc(x_vec, y_vec, V_fld);
colorbar;  axis equal tight;  set(gca, 'YDir', 'normal');
title('Vertical Displacement v(x,y)  [Block AMG]', 'FontSize', 11);
xlabel('x');  ylabel('y');
