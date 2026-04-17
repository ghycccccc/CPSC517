function [A, b, grid_meta] = build_scalar_diffusion_2d(nx, ny, a, f, bc, opts)
% BUILD_SCALAR_DIFFUSION_2D
% Assemble the interior-node finite-difference system for
%
%   -div(a(x,y) grad u) = f
%
% on a rectangular domain with scalar heterogeneous coefficient a.
%
% The discretization is written in divergence form using face coefficients,
% which preserves symmetry for Dirichlet problems and is a better fit for
% scalar AMG than an expanded product-rule stencil.
%
% INPUTS:
%   nx, ny  - number of interior nodes in x and y
%   a       - coefficient field on interior nodes, size [ny x nx]
%   f       - source field on interior nodes, size [ny x nx]
%   bc      - boundary condition struct with fields left/right/bottom/top
%             each containing:
%               .type  = 'dirichlet' or 'neumann'
%               .value = scalar, vector, or function handle
%   opts    - optional settings:
%               .xlim = [xmin xmax], default [0 1]
%               .ylim = [ymin ymax], default [0 1]
%               .face_average = 'harmonic' or 'arithmetic', default harmonic
%
% OUTPUTS:
%   A         - sparse matrix of size [nx*ny, nx*ny]
%   b         - load vector of size [nx*ny, 1]
%   grid_meta - grid metadata for downstream solvers/plots

    if nargin < 6, opts = struct(); end
    if nargin < 5 || isempty(bc), bc = default_bc(); end

    if ~isfield(opts, 'xlim'), opts.xlim = [0, 1]; end
    if ~isfield(opts, 'ylim'), opts.ylim = [0, 1]; end
    if ~isfield(opts, 'face_average'), opts.face_average = 'harmonic'; end

    assert(all(size(a) == [ny, nx]), 'a must be [ny x nx].');
    assert(all(size(f) == [ny, nx]), 'f must be [ny x nx].');

    if ~isfield(bc, 'left'),   bc.left   = struct('type', 'dirichlet', 'value', 0); end
    if ~isfield(bc, 'right'),  bc.right  = struct('type', 'dirichlet', 'value', 0); end
    if ~isfield(bc, 'bottom'), bc.bottom = struct('type', 'dirichlet', 'value', 0); end
    if ~isfield(bc, 'top'),    bc.top    = struct('type', 'dirichlet', 'value', 0); end

    xmin = opts.xlim(1);
    xmax = opts.xlim(2);
    ymin = opts.ylim(1);
    ymax = opts.ylim(2);

    hx = (xmax - xmin) / (nx + 1);
    hy = (ymax - ymin) / (ny + 1);

    x_vec = linspace(xmin + hx, xmax - hx, nx);
    y_vec = linspace(ymin + hy, ymax - hy, ny);

    getNode = @(i, j) (j - 1) * nx + i;

    east_face = interior_face_average(a(:, 1:end-1), a(:, 2:end), opts.face_average);
    north_face = interior_face_average(a(1:end-1, :), a(2:end, :), opts.face_average);

    west_bdry = a(:, 1);
    east_bdry = a(:, end);
    south_bdry = a(1, :);
    north_bdry = a(end, :);

    est_nnz = 5 * nx * ny;
    I = zeros(est_nnz, 1);
    J = zeros(est_nnz, 1);
    V = zeros(est_nnz, 1);
    ptr = 0;
    b = reshape(f.', [], 1);

    for j = 1:ny
        y = y_vec(j);
        for i = 1:nx
            x = x_vec(i);
            row = getNode(i, j);
            diag_val = 0;

            if i > 1
                a_w = east_face(j, i - 1);
                diag_val = diag_val + a_w / hx^2;
                ptr = ptr + 1;
                I(ptr) = row; J(ptr) = getNode(i - 1, j); V(ptr) = -a_w / hx^2;
            else
                a_w = west_bdry(j);
                [diag_add, rhs_add] = boundary_contribution(a_w, hx, bc.left, y);
                diag_val = diag_val + diag_add;
                b(row) = b(row) + rhs_add;
            end

            if i < nx
                a_e = east_face(j, i);
                diag_val = diag_val + a_e / hx^2;
                ptr = ptr + 1;
                I(ptr) = row; J(ptr) = getNode(i + 1, j); V(ptr) = -a_e / hx^2;
            else
                a_e = east_bdry(j);
                [diag_add, rhs_add] = boundary_contribution(a_e, hx, bc.right, y);
                diag_val = diag_val + diag_add;
                b(row) = b(row) + rhs_add;
            end

            if j > 1
                a_s = north_face(j - 1, i);
                diag_val = diag_val + a_s / hy^2;
                ptr = ptr + 1;
                I(ptr) = row; J(ptr) = getNode(i, j - 1); V(ptr) = -a_s / hy^2;
            else
                a_s = south_bdry(i);
                [diag_add, rhs_add] = boundary_contribution(a_s, hy, bc.bottom, x);
                diag_val = diag_val + diag_add;
                b(row) = b(row) + rhs_add;
            end

            if j < ny
                a_n = north_face(j, i);
                diag_val = diag_val + a_n / hy^2;
                ptr = ptr + 1;
                I(ptr) = row; J(ptr) = getNode(i, j + 1); V(ptr) = -a_n / hy^2;
            else
                a_n = north_bdry(i);
                [diag_add, rhs_add] = boundary_contribution(a_n, hy, bc.top, x);
                diag_val = diag_val + diag_add;
                b(row) = b(row) + rhs_add;
            end

            ptr = ptr + 1;
            I(ptr) = row;
            J(ptr) = row;
            V(ptr) = diag_val;
        end
    end

    A = sparse(I(1:ptr), J(1:ptr), V(1:ptr), nx * ny, nx * ny);

    grid_meta = struct();
    grid_meta.nx = nx;
    grid_meta.ny = ny;
    grid_meta.hx = hx;
    grid_meta.hy = hy;
    grid_meta.xlim = opts.xlim;
    grid_meta.ylim = opts.ylim;
    grid_meta.x_vec = x_vec;
    grid_meta.y_vec = y_vec;
    grid_meta.domain = [xmin, xmax, ymin, ymax];
end


function bc = default_bc()
    bc.left   = struct('type', 'dirichlet', 'value', 0);
    bc.right  = struct('type', 'dirichlet', 'value', 0);
    bc.bottom = struct('type', 'dirichlet', 'value', 0);
    bc.top    = struct('type', 'dirichlet', 'value', 0);
end


function avg = interior_face_average(aL, aR, mode)
    switch lower(mode)
        case 'harmonic'
            denom = aL + aR;
            avg = 2 * aL .* aR ./ denom;
            zero_mask = abs(denom) < 1e-14;
            avg(zero_mask) = 0;
        case 'arithmetic'
            avg = 0.5 * (aL + aR);
        otherwise
            error('Unknown face averaging mode: %s', mode);
    end
end


function [diag_add, rhs_add] = boundary_contribution(a_face, h, bc_side, coord)
    bc_type = lower(bc_side.type);
    bc_val = evaluate_bc_value(bc_side.value, coord);

    switch bc_type
        case 'dirichlet'
            diag_add = a_face / h^2;
            rhs_add = a_face * bc_val / h^2;
        case 'neumann'
            diag_add = 0;
            rhs_add = a_face * bc_val / h;
        otherwise
            error('Unsupported boundary type: %s', bc_side.type);
    end
end


function val = evaluate_bc_value(spec, coord)
    if isa(spec, 'function_handle')
        val = spec(coord);
    elseif isscalar(spec)
        val = spec;
    else
        error('Boundary values must be scalar or function handle for this builder.');
    end
end
