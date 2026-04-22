# CPSC 517 Final Project — Block AMG for 2D Heterogeneous Linear Elasticity

This repository contains the MATLAB implementation and experiments for the final project report on block algebraic multigrid (AMG) preconditioning applied to the 2D heterogeneous Navier-Cauchy system of linear elasticity.

**Disclamer:** this code repository is completed with the help of generative AI.

---

## Repository Layout

```
Final/
├── CLAUDE.md                  ← session knowledge base
├── README.md                  ← this file
├── figures/                   ← auto-saved PNG outputs (created on first run)
├── report/
│   ├── Report.md              ← report structure plan
│   └── report.tex             ← compilable LaTeX report
└── scripts/                   ← all MATLAB source files
```

---

## Core Shared Utility

### `build_sweep_matrices.m`

Shared by all three multigrid implementations. Precomputes factored forward and backward Gauss-Seidel sweep matrices for a given matrix `A` and DOF-per-node count `d`.

**Signature:**
```matlab
[M_fwd_fac, M_bwd_fac, AmMfwd, AmMbwd] = build_sweep_matrices(A, dof_per_node)
```

**Key behaviour:**
- For `dof_per_node = 1` (scalar GS): `M_fwd = tril(A)`, `M_bwd = triu(A)`.
- For `dof_per_node = 2` (block GS): `M_fwd` includes all entries where the row-node index ≥ col-node index (i.e., ⌈p/2⌉ ≥ ⌈q/2⌉), capturing within-node (u,v) coupling in the forward sweep.
- Both cases cache LU factorizations via MATLAB's `decomposition(M, 'lu')`, so each subsequent sweep costs one sparse mat-vec plus one triangular solve with zero refactorization overhead.
- The complement matrices `AmMfwd = A − M_fwd` and `AmMbwd = A − M_bwd` enable the vectorized sweep formula `x = M_fac \ (b − AmM * x)`.

---

## PDE Assembler

### `build_navier_cauchy_heterogeneous.m`

Assembles the stiffness matrix `K` and load vector `b` for the 2D heterogeneous Navier-Cauchy equations on a uniform `nx × ny` grid with spacing `h`.

**Signature:**
```matlab
[K, b] = build_navier_cauchy_heterogeneous(nx, ny, h, lambda, mu, fx, fy, opts)
```

**Key behaviour:**
- Accepts spatially varying `lambda` and `mu` as `[ny × nx]` arrays.
- Implements the strong-form FD expansion via the product rule, producing both diagonal-neighbour u-v coupling (from the mixed derivative term) and axis-aligned u-v coupling (from gradient-correction terms proportional to ∇λ and ∇μ).
- For heterogeneous materials, `K` is non-symmetric: `K[u_{i,j}, v_{i+1,j}]` involves ∂μ/∂y while `K[v_{i+1,j}, u_{i,j}]` involves ∂λ/∂y, which differ in general.
- Supports `opts.dof_ordering = 'interleaved'` (default, u₁v₁u₂v₂…) or `'component'` (u₁…uN v₁…vN).

---

## Scalar AMG

### `amg_setup.m`

Builds a scalar Ruge-Stüben AMG hierarchy for the `2N × 2N` system `K`.

**Signature:**
```matlab
hierarchy = amg_setup(A, theta, max_levels, coarse_threshold)
```

**Internal pipeline:**
1. `compute_strong_dependence`: identifies strong connections using only negative off-diagonal entries (M-matrix assumption), with threshold `θ`.
2. `cf_splitting`: greedy C/F assignment + H-1 enforcement pass.
3. `build_interpolation`: classical RS scalar weights `ω_ij = −(a_{ij} + twice-removed) / (a_{ii} + Σ_weak a_{ik})`.
4. Galerkin coarse operator `A_c = R A P`.
5. `build_sweep_matrices(A, 1)` called at every level to store vectorized scalar GS sweep objects.

**Limitation on the Navier-Cauchy system:** Positive off-diagonal u-v coupling entries are ignored (misclassified as weak), and u_i/v_i of the same node can be assigned to opposite C/F classes — causing operator complexity ≈14–15× and preconditioner divergence.

### `amg_vcycle.m`

Applies one V-cycle of scalar AMG.

```matlab
x = amg_vcycle(hierarchy, lev, b, x, nu1, nu2)
```

Pre- and post-smoothing use the vectorized formula:
```matlab
x = hierarchy{lev}.M_fwd_fac \ (b - hierarchy{lev}.AmMfwd * x);  % forward
x = hierarchy{lev}.M_bwd_fac \ (b - hierarchy{lev}.AmMbwd * x);  % backward
```

### `amg_preconditioner.m`

Wrapper that applies one full scalar AMG V-cycle from a zero initial guess.

```matlab
z = amg_preconditioner(hierarchy, r, nu1, nu2)
```

---

## Block AMG — Proportional Weights + SA Smoothing

### `block_amg_setup.m`

Builds a node-level block AMG hierarchy where each physical node's (u,v) pair is treated as an indivisible unit throughout coarsening. This is the primary method (M3/M4 in the ablation study).

**Signature:**
```matlab
hierarchy = block_amg_setup(K, theta, max_levels, coarse_threshold)
```

**Internal pipeline:**
1. `node_strength_matrix`: computes `S(i,j) = ‖K_block(i,j)‖_F` (Frobenius norm of the 2×2 off-diagonal block), always non-negative — no sign filtering needed.
2. `node_strong_dependence` + `cf_splitting`: same greedy + H-1 algorithm applied at node level; both u_i and v_i are always coarsened together.
3. `block_prolongation`: tentative prolongation with scalar proportional weights `ω_ij = s_ij / Σ s_ik`, inserted as `ω_ij · I₂` blocks. Satisfies partition of unity (rigid-body translation preserved exactly).
4. `smooth_prolongation`: SA-style Jacobi smoothing step `P = P_tent − ω · D_block⁻¹ · K · P_tent`, with `ω = 4/(3ρ)` estimated by 10 power iterations. Enforces the AMG interpolation property `K · P · e_c ≈ 0` for smooth error.
5. Galerkin coarse operator `K_c = P^T K P`.
6. `build_sweep_matrices` called twice per level: once with `dof_per_node=2` (block GS fields) and once with `dof_per_node=1` (scalar GS fields), so the same hierarchy serves either smoother.

**Hierarchy complexity (64×64, 3× contrast):** OC ≈ 2.2×.

---

## Block AMG — Ruge-Stüben Block Weights

### `block_amg_setup_rs.m`

Alternative block AMG hierarchy using direct Ruge-Stüben block interpolation weights without SA prolongation smoothing. Used as method M2 in the ablation study.

**Signature:**
```matlab
hierarchy = block_amg_setup_rs(K, theta, max_levels, coarse_threshold)
```

**Identical to `block_amg_setup` except for the prolongation.**

`block_prolongation_rs` computes a full **2×2 matrix** weight for each F-to-C coupling:

```
ω_ij = D_i⁻¹ · (K_ij + T_ij + E_per_j)
```

where:
- **D_i** = `K_block(i,i) + Σ_{k∈W_i} K_block(i,k)` (diagonal + weak neighbors, inverted as 2×2)
- **K_ij**: direct 2×2 coupling block
- **T_ij** (twice-removed): `Σ_{k∈T_i} K_ik · ‖K_kj‖_F / Σ_{l∈I_i∩I_k} ‖K_kl‖_F` — distributes indirect coupling from strong F-neighbors proportionally by Frobenius norm
- **E_per_j** (simple average): `(1/|I_i|) · Σ_{k∈E_i} K_ik` — equal share for isolated F-neighbors with no common C-neighbor

The 2×2 matrix weight (vs. scalar×I₂ in the proportional approach) can represent anisotropic u-v coupling in F-node interpolation. No `smooth_prolongation` call — the RS formula attempts to satisfy the local equilibrium equation in block form directly.

Stores the same sweep-matrix fields as `block_amg_setup`, so `block_amg_vcycle` works unchanged with either hierarchy.

---

## Shared Block AMG V-Cycle

### `block_amg_vcycle.m`

Shared V-cycle for both block AMG hierarchies. Supports an optional seventh argument to select the smoother type.

```matlab
x = block_amg_vcycle(hierarchy, lev, b, x, nu1, nu2)           % block GS (default)
x = block_amg_vcycle(hierarchy, lev, b, x, nu1, nu2, 'scalar') % scalar GS
x = block_amg_vcycle(hierarchy, lev, b, x, nu1, nu2, 'block')  % block GS (explicit)
```

- `'block'`: uses `M_fwd_fac`, `M_bwd_fac`, `AmMfwd`, `AmMbwd` (node-level block triangular)
- `'scalar'`: uses `M_fwd_scalar_fac`, `M_bwd_scalar_fac`, `AmMfwd_scalar`, `AmMbwd_scalar` (standard tril/triu)

The smoother argument is forwarded recursively to all coarse levels. The V-cycle structure is otherwise identical to `amg_vcycle.m`.

---

## Geometric Multigrid (Block)

### `geomg_setup.m`

Builds a geometric multigrid hierarchy using bilinear interpolation on the Cartesian grid.

```matlab
hierarchy = geomg_setup(A, grid_meta, max_levels, coarse_threshold)
```

- `grid_meta` must contain `.nx`, `.ny`, and optionally `.dof_per_node` (default 1).
- For `dof_per_node = 2`: block prolongation `P = kron(P_scalar, speye(2))`, so both displacement components at each node receive identical interpolation weights from the coarse node.
- Coarse grid halves each dimension: `nx_c = floor((nx_f − 1) / 2)`.
- Restriction: `R = 0.25 · P^T` (scaled full-weighting).
- Coarse operator: Galerkin `A_c = R A P`.
- `build_sweep_matrices(A_c, dof_per_node)` called at each new level.
- Stopping criterion: `n_lev / dof_per_node ≤ coarse_threshold` or `min(nx,ny) ≤ 2`.

### `geomg_preconditioner.m`

Applies one block geometric MG V-cycle from zero initial guess.

```matlab
z = geomg_preconditioner(hierarchy, r, nu1, nu2)
```

The internal `geomg_vcycle` uses the same vectorized sweep formula as `amg_vcycle` and `block_amg_vcycle`, reading precomputed sweep objects from each hierarchy level.

---

## Benchmark Infrastructure

### `run_preconditioner_experiment.m`

Unified benchmark harness that assembles a problem, runs multiple preconditioned GMRES solves, and saves figures.

**Key features:**
- Accepts a `problem` struct (matrix `A`, rhs `b`, grid metadata, coefficient field for visualization)
- Accepts `pc_opts.methods` specifying which preconditioners to run: `'plain'`, `'jacobi'`, `'gauss-seidel'`, `'sor'`, `'ilu'`, `'geomg'`, `'amg'`, `'block-jacobi'`, `'block-amg'`
- Calls `amg_setup` for `'amg'`, `geomg_setup` for `'geomg'`, `block_amg_setup` for `'block-amg'`
- Reports iteration count, relative residual, setup time, solve time
- Saves log-log convergence figures and problem setup figures to `../figures/`

Used by the scaling (`exp_grid_scaling.m`) and contrast sweep (`exp_contrast_sweep.m`) experiments.

---

## Experiment Scripts

### `exp_grid_scaling.m` — Experiment 3: Grid Resolution Scaling

Tests all preconditioners at four grid sizes (16×16, 32×32, 64×64, 128×128) with fixed 3× material contrast.

**Outputs:** Iteration count and total time tables; figures saved to `figures/`.

**Key result:** Block AMG achieves near-mesh-independent iteration counts (9→12 over 64× DOF increase) but is the most expensive method in wall time (14.33 s at 128×128 vs. 0.57 s for Geometric MG) due to hierarchy setup cost.

### `exp_contrast_sweep.m` — Experiment 4: Material Contrast Sweep

Tests all preconditioners at five contrast levels (1×, 2×, 5×, 10×, 20×) on a fixed 64×64 grid.

**Outputs:** Iteration count and time tables; figures saved to `figures/`.

**Key result:** Block AMG excels at low-to-moderate contrast (9–10 iterations up to 5×) but fails at 20× contrast (14800 iterations, effectively diverges) due to the non-symmetric K produced by the strong-form FD discretisation, whose asymmetry grows with material gradients.

### `exp_amg_ablation.m` — Experiment 5: AMG Design Ablation

Isolates the contribution of two design choices — prolongation strategy and smoother type — on a fixed 64×64, 3× contrast problem.

**Four methods tested:**

| ID | Method | Hierarchy | Smoother |
|----|--------|-----------|----------|
| M1 | Scalar AMG | `amg_setup` | Scalar GS |
| M2 | Block AMG RS | `block_amg_setup_rs` | Block GS |
| M3 | Block AMG prop+SA | `block_amg_setup` | Scalar GS |
| M4 | Block AMG prop+SA | `block_amg_setup` | Block GS |

M3 and M4 share one hierarchy; M2 uses a separate RS hierarchy.

**Outputs (4 figures):**
- Log-log GMRES residual histories for M1–M4
- Iteration count bar chart
- Stacked setup/solve time bar chart (log y-axis)
- Hierarchy complexity: nnz and DOF count per level for all three hierarchies

**Key result:** M3/M4 (prop+SA) both converge in 9 iterations; M1 (scalar AMG) diverges (residual stays at 2.89×10²); M2 (RS block) approaches but does not meet tolerance (residual 1.48×10⁻⁵ after 100 iterations). Scalar vs. block GS smoother has negligible effect on iteration count at 3× contrast.

### `exp_baseline_comparison.m` — Experiment 1

Compares all preconditioners on a moderate baseline problem. Used for the initial preconditioner comparison in the report introduction. Configuration mirrors `run_preconditioner_experiment.m` defaults.

---

## Miscellaneous / Demo Scripts (Not Reflected in Final Report)

The following scripts were used for early development, debugging, and minimal case demonstrations in the midterm presentation. They are not part of the final experimental pipeline:

- **`demo_amg_elasticity.m`**, **`demo_block_amg_elasticity.m`**: minimal three-way convergence demonstrations on small (30×30) grids
- **`demo_amg_poisson_dirichlet.m`**: simple scalar Poisson demo for AMG sanity check
- **`demo_diffusion_heterogeneous_benchmark.m`**: partial implementation; scalar diffusion channel-barrier case (solve block commented out)
- **`demo_lognormal_permeability.m`**: 80×80 log-normal scalar diffusion benchmark with sigma sweep
- **`build_scalar_diffusion_2d.m`**, **`make_diffusion_benchmark_case.m`**: scalar diffusion assembler and benchmark case factory; not used in elasticity experiments
- **`solve_with_amg_gmres.m`**, **`solve_with_block_amg_gmres.m`**, **`amg_vcycle_and_gmres.m`**: standalone GMRES drivers used in early development; superseded by `run_preconditioner_experiment.m` and the ablation harness
- **`tmp_trace_amg_issue.m`**, **`tmp_compare_cases.m`**: temporary debugging scripts

---

## Quick Start

```matlab
cd('Final/scripts');

% --- Scaling experiment (Experiment 3) ---
exp_grid_scaling

% --- Contrast sweep (Experiment 4) ---
exp_contrast_sweep

% --- AMG design ablation (Experiment 5) ---
exp_amg_ablation
```

Figures are saved automatically to `../figures/` as PNG files at 150 DPI.

**AMG parameters** used throughout: `theta = 0.25`, `max_levels = 8`, `coarse_threshold = 40`, `nu1 = nu2 = 1`.
