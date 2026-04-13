% =========================================================================
%  demo_amg_elasticity.m
%  Quick demo: AMG-preconditioned GMRES on 2D heterogeneous linear
%  elasticity (Navier-Cauchy) with a stiff circular inclusion.
%
%  Requires:
%    build_navier_cauchy_heterogeneous.m
%    amg_setup.m
%    amg_vcycle_and_gmres.m   (contains amg_vcycle, amg_preconditioner,
%                               solve_with_amg_gmres, gauss_seidel)
% =========================================================================
clear; clc; close all;

fprintf('=================================================\n');
fprintf(' AMG-Preconditioned GMRES Demo\n');
fprintf(' 2D Heterogeneous Navier-Cauchy (Linear Elasticity)\n');
fprintf('=================================================\n\n');

% -------------------------------------------------------------------------
% 1. GRID AND MATERIAL SETUP
% -------------------------------------------------------------------------
nx = 30;          % grid points in x
ny = 30;          % grid points in y
h  = 1 / (nx+1); % grid spacing (unit domain)

% Coordinate arrays (node centers, Matlab convention: row=y, col=x)
x_vec = linspace(h, 1-h, nx);
y_vec = linspace(h, 1-h, ny);
[X, Y] = meshgrid(x_vec, y_vec);   % both [ny x nx]

% ---- Material: soft background with stiff circular inclusion ----
% Background: aluminum-like   (mu=26, lambda=51)
% Inclusion:  steel-like      (mu=77, lambda=115)
% Contrast ratio ~ 3x in mu, which is enough to break homogeneous symmetry

mu_bg  = 26;   lambda_bg  = 51;
mu_inc = 77;   lambda_inc = 115;

% Circular inclusion centered at (0.5, 0.5) with radius 0.2
r_inc = 0.2;
cx    = 0.5;
cy    = 0.5;
in_inclusion = ((X - cx).^2 + (Y - cy).^2) <= r_inc^2;

% Build material arrays [ny x nx]
mu_arr     = mu_bg     * ones(ny, nx);
lambda_arr = lambda_bg * ones(ny, nx);
mu_arr    (in_inclusion) = mu_inc;
lambda_arr(in_inclusion) = lambda_inc;

fprintf('Material setup:\n');
fprintf('  Background: mu=%.0f, lambda=%.0f\n', mu_bg, lambda_bg);
fprintf('  Inclusion:  mu=%.0f, lambda=%.0f\n', mu_inc, lambda_inc);
fprintf('  Contrast ratio (mu): %.1fx\n\n', mu_inc/mu_bg);

% ---- Body forces: uniform downward load (gravity-like) ----
fx = zeros(ny, nx);
fy = -ones(ny, nx);    % downward force in y-direction

% -------------------------------------------------------------------------
% 2. ASSEMBLE SYSTEM  K * x = b
% -------------------------------------------------------------------------
fprintf('Assembling K and b...\n');
t_assemble = tic;
[K, b] = build_navier_cauchy_heterogeneous(nx, ny, h, ...
                                            lambda_arr, mu_arr, fx, fy);
t_assemble = toc(t_assemble);

n_dofs = size(K, 1);
fprintf('  System size:     %d x %d\n', n_dofs, n_dofs);
fprintf('  Nonzeros in K:   %d\n', nnz(K));
fprintf('  Assembly time:   %.3f seconds\n\n', t_assemble);

% -------------------------------------------------------------------------
% 3. SOLVE WITH AMG-PRECONDITIONED GMRES
% -------------------------------------------------------------------------

% AMG options
amg_opts.theta            = 0.25;   % strong dependence threshold
amg_opts.max_levels       = 8;      % maximum hierarchy levels
amg_opts.coarse_threshold = 40;     % direct solve below this size
amg_opts.nu1              = 1;      % pre-smoothing sweeps
amg_opts.nu2              = 1;      % post-smoothing sweeps

% GMRES options
gmres_opts.tol      = 1e-8;    % relative residual tolerance
gmres_opts.max_iter = 200;     % maximum iterations
gmres_opts.restart  = 30;      % Krylov restart

fprintf('--- AMG-Preconditioned GMRES ---\n');
[x_amg, resvec_amg, iter_amg] = solve_with_amg_gmres(K, b, amg_opts, gmres_opts);
fprintf('\n');

% -------------------------------------------------------------------------
% 4. COMPARE: UNPRECONDITIONED GMRES  (baseline)
% -------------------------------------------------------------------------
fprintf('--- Unpreconditioned GMRES (baseline) ---\n');
t_plain = tic;
[x_plain, flag_plain, ~, iter_plain, resvec_plain] = gmres( ...
    K, b, gmres_opts.restart, gmres_opts.tol, gmres_opts.max_iter);
t_plain = toc(t_plain);

if numel(iter_plain) == 2
    iter_plain_total = (iter_plain(1)-1)*gmres_opts.restart + iter_plain(2);
else
    iter_plain_total = iter_plain;
end
fprintf('  Converged flag:  %d\n', flag_plain);
fprintf('  Iterations:      %d\n', iter_plain_total);
fprintf('  Solve time:      %.3f seconds\n\n', t_plain);

% -------------------------------------------------------------------------
% 5. VERIFY SOLUTION ACCURACY
% -------------------------------------------------------------------------
res_amg   = norm(b - K*x_amg)   / norm(b);
res_plain = norm(b - K*x_plain) / norm(b);

fprintf('Solution verification:\n');
fprintf('  AMG-GMRES   relative residual: %.2e\n', res_amg);
fprintf('  Plain GMRES relative residual: %.2e\n', res_plain);
fprintf('  Solution difference ||x_amg - x_plain|| / ||x_plain||: %.2e\n\n', ...
        norm(x_amg - x_plain) / norm(x_plain));

% -------------------------------------------------------------------------
% 6. PLOTS
% -------------------------------------------------------------------------

% ---- Figure 1: Convergence comparison ----
figure('Name', 'Convergence Comparison', 'Position', [100 100 700 450]);

semilogy(0:length(resvec_amg)-1,   resvec_amg   / resvec_amg(1),   'b-',  'LineWidth', 2);
hold on;
semilogy(0:length(resvec_plain)-1, resvec_plain / resvec_plain(1), 'r--', 'LineWidth', 2);
yline(gmres_opts.tol, 'k:', 'LineWidth', 1.5);

xlabel('GMRES Iteration');
ylabel('Relative Residual Norm');
title('AMG-Preconditioned vs Unpreconditioned GMRES');
legend('AMG-GMRES', 'Plain GMRES', 'Tolerance', 'Location', 'southwest');
grid on;
xlim([0, max(length(resvec_amg), length(resvec_plain))]);
ylim([1e-10, 2]);

% ---- Figure 2: Material distribution ----
figure('Name', 'Problem Setup', 'Position', [820 100 900 380]);

subplot(1,2,1);
imagesc(x_vec, y_vec, mu_arr);
colorbar; axis equal tight; set(gca,'YDir','normal');
title('Shear Modulus \mu(x,y)');
xlabel('x'); ylabel('y');
colormap(gca, 'hot');

subplot(1,2,2);
imagesc(x_vec, y_vec, lambda_arr);
colorbar; axis equal tight; set(gca,'YDir','normal');
title('Lame Parameter \lambda(x,y)');
xlabel('x'); ylabel('y');
colormap(gca, 'hot');

% ---- Figure 3: Displacement field ----
% Unpack solution into u and v fields [ny x nx]
u_vec = zeros(2*nx*ny, 1);
v_vec = zeros(2*nx*ny, 1);

getNode = @(i,j) (j-1)*nx + i;
getUdof = @(n)   2*n - 1;
getVdof = @(n)   2*n;

U = zeros(ny, nx);
V_disp = zeros(ny, nx);
for j = 1:ny
    for i = 1:nx
        nd     = getNode(i,j);
        U(j,i) = x_amg(getUdof(nd));
        V_disp(j,i) = x_amg(getVdof(nd));
    end
end

figure('Name', 'Displacement Field', 'Position', [100 560 900 380]);

subplot(1,2,1);
imagesc(x_vec, y_vec, U);
colorbar; axis equal tight; set(gca,'YDir','normal');
title('Horizontal Displacement u(x,y)');
xlabel('x'); ylabel('y');

subplot(1,2,2);
imagesc(x_vec, y_vec, V_disp);
colorbar; axis equal tight; set(gca,'YDir','normal');
title('Vertical Displacement v(x,y)');
xlabel('x'); ylabel('y');

% -------------------------------------------------------------------------
% 7. SUMMARY TABLE
% -------------------------------------------------------------------------
fprintf('=========================================\n');
fprintf('  SUMMARY\n');
fprintf('=========================================\n');
fprintf('  Grid:           %d x %d  (%d DOFs)\n', nx, ny, n_dofs);
fprintf('  Contrast ratio: %.0fx\n', mu_inc/mu_bg);
fprintf('%-20s %10s %10s\n', 'Method', 'Iterations', 'Rel.Resid.');
fprintf('%-20s %10d %10.2e\n', 'AMG-GMRES',   iter_amg,         res_amg);
fprintf('%-20s %10d %10.2e\n', 'Plain GMRES', iter_plain_total, res_plain);
fprintf('=========================================\n');
