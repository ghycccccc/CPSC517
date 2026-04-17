function hierarchy = geomg_setup(A, grid_meta, max_levels, coarse_threshold)
% GEOMG_SETUP
% Build a simple geometric multigrid hierarchy on a rectangular grid using
% bilinear interpolation, scaled full-weighting restriction, and Galerkin
% coarse operators.

    if nargin < 3 || isempty(max_levels), max_levels = 8; end
    if nargin < 4 || isempty(coarse_threshold), coarse_threshold = 40; end

    nx = grid_meta.nx;
    ny = grid_meta.ny;

    hierarchy = {};
    hierarchy{1} = struct('A', A, 'P', [], 'R', [], 'nx', nx, 'ny', ny);

    for lev = 1:max_levels - 1
        A_lev = hierarchy{lev}.A;
        n_lev = size(A_lev, 1);
        nx_f = hierarchy{lev}.nx;
        ny_f = hierarchy{lev}.ny;

        if n_lev <= coarse_threshold || min(nx_f, ny_f) <= 2
            break;
        end

        nx_c = max(1, floor((nx_f - 1) / 2));
        ny_c = max(1, floor((ny_f - 1) / 2));

        if nx_c >= nx_f || ny_c >= ny_f
            break;
        end

        P = kron(build_interp_1d(ny_f, ny_c), build_interp_1d(nx_f, nx_c));
        R = 0.25 * P';
        A_c = R * A_lev * P;

        hierarchy{lev}.P = P;
        hierarchy{lev}.R = R;
        hierarchy{lev + 1} = struct('A', A_c, 'P', [], 'R', [], 'nx', nx_c, 'ny', ny_c);
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
            rows = [rows; i]; %#ok<AGROW>
            cols = [cols; 1]; %#ok<AGROW>
            vals = [vals; weight]; %#ok<AGROW>
            continue;
        end

        if x >= xc(end)
            weight = (1 - x) / (1 - xc(end));
            rows = [rows; i]; %#ok<AGROW>
            cols = [cols; nc]; %#ok<AGROW>
            vals = [vals; weight]; %#ok<AGROW>
            continue;
        end

        k = find(xc <= x, 1, 'last');
        if xc(k) == x
            rows = [rows; i]; %#ok<AGROW>
            cols = [cols; k]; %#ok<AGROW>
            vals = [vals; 1]; %#ok<AGROW>
        else
            xL = xc(k);
            xR = xc(k + 1);
            wL = (xR - x) / (xR - xL);
            wR = (x - xL) / (xR - xL);
            rows = [rows; i; i]; %#ok<AGROW>
            cols = [cols; k; k + 1]; %#ok<AGROW>
            vals = [vals; wL; wR]; %#ok<AGROW>
        end
    end

    P = sparse(rows, cols, vals, nf, nc);
end
