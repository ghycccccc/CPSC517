This is the project repository for CPSC 517 midterm project. Executable MATLAB scripts are under `scripts` directory:

- `build_navier_cauchy(nx, ny, h, lambda, mu, fx, fy, ordering)`: constructs `K` matrix and problem `b` for 2D Navier Cauchy equation. `ordering` is either `node` or `component`;
- `build_navier_cauchy_3d(nx, ny, nz, h, lambda, mu, fx, fy, fz)`: constructs `K` matrix and problem `b` for 3D Navier Cauchy equation (not used in project);
- `compare_gmres_preconditioners(K, b)`: solves `Kx=b` problem with GMRES, augmented with different preconditioners, and give convergence results;
- `dirichlet_demo_case(nx, ny, h, mu, lambda)`: constructs a `K` and `b` problem with Dirichlet boundary conditions;
- `demo.m`: simple demo case with `Nx = ny = 50` and Lame parameters for 3 different materials;

**Disclaimer:** this project is completed with the help of generative AI.