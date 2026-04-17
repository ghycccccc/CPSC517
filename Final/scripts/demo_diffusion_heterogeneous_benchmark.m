% =========================================================================
%  demo_diffusion_heterogeneous_benchmark.m
%  Main scalar diffusion benchmark for comparing plain GMRES, ILU,
%  geometry MG, and AMG on a heterogeneous coefficient field.
% =========================================================================
clear; clc; close all;

nx = 127;
ny = 127;

nx = 90;
ny = 90;

case_opts = struct();
case_opts.contrast = 100;
case_opts.forcing = 'ones';
case_opts.bc_variant = 'dirichlet_all';
case_opts.channel_width = 0.08;
case_opts.barrier_width = 0.04;

case_data = make_diffusion_benchmark_case('channel_barrier', nx, ny, case_opts);
[A, b, grid_meta] = build_scalar_diffusion_2d(nx, ny, ...
    case_data.a, case_data.f, case_data.bc, struct('face_average', 'harmonic'));

problem.A = A;
problem.b = b;
problem.name = sprintf('Heterogeneous Diffusion Benchmark (%d x %d)', nx, ny);
problem.grid_meta = grid_meta;
problem.case_meta = case_data.case_meta;

gmres_opts.tol      = 1e-8;
gmres_opts.max_iter = 200;
gmres_opts.restart  = 40;

amg_opts.theta            = 0.25;
amg_opts.max_levels       = 8;
amg_opts.coarse_threshold = 60;
amg_opts.nu1              = 1;
amg_opts.nu2              = 1;

pc_opts.methods   = {'plain', 'ilu', 'geomg', 'amg'};
pc_opts.sor_omega = 1.2;
pc_opts.ilu_setup = struct('type', 'nofill');

results = run_preconditioner_experiment(problem, gmres_opts, amg_opts, pc_opts);
