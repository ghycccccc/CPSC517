% =========================================================================
%  exp_baseline_comparison.m
%
%  Experiment 1 — Baseline preconditioner comparison for 2D heterogeneous
%  Navier-Cauchy (linear elasticity).
%
%  Problem:
%    128×128 uniform grid, h = 1/129.
%    Soft background (aluminium-like): mu=26, lambda=51.
%    Stiff centred circular inclusion (steel-like): mu=77, lambda=115 (~3× contrast).
%    Uniform downward body force fy = -1.  Zero Dirichlet BCs on all walls.
%
%  Methods compared:
%    1. Plain GMRES              — unpreconditioned baseline
%    2. ILU(0)                   — classical sparse factorisation preconditioner
%    3. Gauss-Seidel             — point GS (forward-triangular solve each step)
%    4. Block Jacobi (2×2)       — per-node diagonal block inverse; captures u-v coupling
%    5. Scalar AMG               — node-agnostic Ruge-Stüben AMG (ignores u-v pairing)
%    6. Block AMG                — node-level coarsening + BGS smoother + SA-AMG smoothing
%
%  Outputs  (auto-saved to ../figures/):
%    *_setup.png                — matrix sparsity pattern (coloured by magnitude)
%    *_hierarchy_complexity.png — nnz and DOF count per AMG level (scalar vs block)
%    *_convergence.png          — GMRES residual history, all methods overlaid
%
%  Prints a summary table with: iterations, true relative residual, setup
%  time, and solve time per method.
% =========================================================================
clear; clc; close all;

% -------------------------------------------------------------------------
% 1. Material and geometry setup
% -------------------------------------------------------------------------
nx = 64;  ny = 64;
h  = 1 / (nx + 1);

x_vec = linspace(h, 1 - h, nx);
y_vec = linspace(h, 1 - h, ny);
[X, Y] = meshgrid(x_vec, y_vec);

mu_bg = 26;   lambda_bg  = 51;
mu_inc = 77;  lambda_inc = 115;
r_inc = 0.2;  cx = 0.5;  cy = 0.5;

in_inc = ((X - cx).^2 + (Y - cy).^2) <= r_inc^2;

mu_arr     = mu_bg     * ones(ny, nx);
lambda_arr = lambda_bg * ones(ny, nx);
mu_arr    (in_inc) = mu_inc;
lambda_arr(in_inc) = lambda_inc;

fx = rand(ny, nx);
fy = rand(ny, nx);

contrast = mu_inc / mu_bg;

% -------------------------------------------------------------------------
% 2. Assemble system
% -------------------------------------------------------------------------
fprintf('Assembling %d×%d Navier-Cauchy system (contrast %.0f×)...\n', ...
        nx, ny, contrast);
asm_opts.dof_ordering = 'node';
[K, b] = build_navier_cauchy_heterogeneous(nx, ny, h, lambda_arr, mu_arr, fx, fy, asm_opts);
fprintf('  DOFs: %d,   nnz(K): %d\n\n', size(K, 1), nnz(K));

% -------------------------------------------------------------------------
% 3. Problem struct
% -------------------------------------------------------------------------
problem.A            = K;
problem.b            = b;
problem.name         = sprintf('Navier-Cauchy %dx%d contrast %.0fx', nx, ny, contrast);
problem.dof_per_node = 2;   % interleaved u,v DOF ordering (required by block methods)

% Grid metadata for geometric MG.
% dof_per_node=2 tells geomg_setup to build P_block = kron(P_scalar, I_2)
% so that each node's (u,v) pair is interpolated by the same bilinear weights.
problem.grid_meta.nx           = nx;
problem.grid_meta.ny           = ny;
problem.grid_meta.hx           = h;
problem.grid_meta.hy           = h;
problem.grid_meta.dof_per_node = 2;

% -------------------------------------------------------------------------
% 4. Solver options
% -------------------------------------------------------------------------
gmres_opts.tol      = 1e-8;
gmres_opts.max_iter = 300;
gmres_opts.restart  = 50;

amg_opts.theta            = 0.25;
amg_opts.max_levels       = 8;
amg_opts.coarse_threshold = 40;
amg_opts.nu1              = 1;
amg_opts.nu2              = 1;

%pc_opts.methods = {'plain', 'ilu', 'gauss-seidel', 'block-jacobi', 'amg', 'block-amg'};
pc_opts.methods = {'plain', 'ilu', 'gauss-seidel', 'block-jacobi', 'geomg', 'block-amg'};

% -------------------------------------------------------------------------
% 5. Run experiment  (figures auto-saved to ../figures/)
% -------------------------------------------------------------------------
results = run_preconditioner_experiment(problem, gmres_opts, amg_opts, pc_opts);

% -------------------------------------------------------------------------
% 6. Summary table
% -------------------------------------------------------------------------
fprintf('\n');
fprintf('================================================================\n');
fprintf('  SUMMARY  (%d×%d grid, %d DOFs, contrast %.0f×)\n', ...
        nx, ny, size(K, 1), contrast);
fprintf('================================================================\n');
fprintf('%-22s %8s %12s %10s %10s\n', 'Method', 'Iters', 'Rel.Resid', 'Setup(s)', 'Solve(s)');
fprintf('%s\n', repmat('-', 1, 66));
for i = 1:numel(results)
    r = results(i);
    if isnan(r.setup_time)
        st = '       N/A';
    else
        st = sprintf('%10.3f', r.setup_time);
    end
    fprintf('%-22s %8d %12.2e %10s %10.3f\n', ...
            r.method, r.iter, r.true_resid, st, r.solve_time);
end
fprintf('================================================================\n');
