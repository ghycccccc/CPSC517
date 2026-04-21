function hierarchy = geomg_setup(A, grid_meta, max_levels, coarse_threshold)
% GEOMG_SETUP
% Build a simple geometric multigrid hierarchy on a rectangular grid using
% bilinear interpolation, scaled full-weighting restriction, and Galerkin
% coarse operators.
%
% For vector-field problems (e.g. 2D elasticity with interleaved u,v DOFs),
% set grid_meta.dof_per_node = 2 (default 1).  The scalar bilinear
% interpolation is then expanded to a block prolongation
%   P_block = kron(P_scalar, speye(dof_per_node))
% so that each physical node's DOFs are interpolated identically.

    if nargin < 3 || isempty(max_levels), max_levels = 8; end
    if nargin < 4 || isempty(coarse_threshold), coarse_threshold = 40; end

    nx = grid_meta.nx;
    ny = grid_meta.ny;

    dof_per_node = 1;
    if isfield(grid_meta, 'dof_per_node')
        dof_per_node = grid_meta.dof_per_node;
    end

    hierarchy = {};
    hierarchy{1} = struct('A', A, 'P', [], 'R', [], 'nx', nx, 'ny', ny);
    [hierarchy{1}.M_fwd_fac, hierarchy{1}.M_bwd_fac, ...
     hierarchy{1}.AmMfwd,    hierarchy{1}.AmMbwd] = ...
        build_sweep_matrices(A, dof_per_node);

    for lev = 1:max_levels - 1
        A_lev = hierarchy{lev}.A;
        n_lev = size(A_lev, 1);
        nx_f = hierarchy{lev}.nx;
        ny_f = hierarchy{lev}.ny;

        % Stopping criterion on physical nodes, not raw DOF count
        if (n_lev / dof_per_node) <= coarse_threshold || min(nx_f, ny_f) <= 2
            break;
        end

        nx_c = max(1, floor((nx_f - 1) / 2));
        ny_c = max(1, floor((ny_f - 1) / 2));

        if nx_c >= nx_f || ny_c >= ny_f
            break;
        end

        % Scalar bilinear interpolation on the node grid
        P_scalar = kron(build_interp_1d(ny_f, ny_c), build_interp_1d(nx_f, nx_c));

        % Expand to block prolongation for multi-DOF problems
        if dof_per_node > 1
            P = kron(P_scalar, speye(dof_per_node));
        else
            P = P_scalar;
        end

        R = 0.25 * P';
        A_c = R * A_lev * P;

        hierarchy{lev}.P = P;
        hierarchy{lev}.R = R;
        hierarchy{lev + 1} = struct('A', A_c, 'P', [], 'R', [], 'nx', nx_c, 'ny', ny_c);
        [hierarchy{lev+1}.M_fwd_fac, hierarchy{lev+1}.M_bwd_fac, ...
         hierarchy{lev+1}.AmMfwd,    hierarchy{lev+1}.AmMbwd] = ...
            build_sweep_matrices(A_c, dof_per_node);
    end
end


function P = build_interp_1d(nf, nc)
    xf = (1:nf)' / (nf + 1);
    xc = (1:nc)' / (nc + 1);

    rows = [];
    cols = [];
    vals = [];

    for i = 1:nf
        x = xf(i);

        if x <= xc(1)
            weight = x / xc(1);
            rows = [rows; i]; 
            cols = [cols; 1]; 
            vals = [vals; weight]; 
            continue;
        end

        if x >= xc(end)
            weight = (1 - x) / (1 - xc(end));
            rows = [rows; i]; 
            cols = [cols; nc]; 
            vals = [vals; weight]; 
            continue;
        end

        k = find(xc <= x, 1, 'last');
        if xc(k) == x
            rows = [rows; i]; 
            cols = [cols; k]; 
            vals = [vals; 1]; 
        else
            xL = xc(k);
            xR = xc(k + 1);
            wL = (xR - x) / (xR - xL);
            wR = (x - xL) / (xR - xL);
            rows = [rows; i; i]; 
            cols = [cols; k; k + 1]; 
            vals = [vals; wL; wR]; 
        end
    end

    P = sparse(rows, cols, vals, nf, nc);
end
