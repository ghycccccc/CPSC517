function case_data = make_diffusion_benchmark_case(case_name, nx, ny, case_opts)
% MAKE_DIFFUSION_BENCHMARK_CASE
% Build coefficient/source data for scalar diffusion benchmarks.
%
% OUTPUT fields:
%   .a          coefficient field [ny x nx]
%   .f          source field [ny x nx]
%   .bc         boundary condition struct
%   .name       descriptive case name
%   .u_exact    optional exact solution [ny x nx], empty otherwise
%   .case_meta  metadata for experiment reporting

    if nargin < 4, case_opts = struct(); end
    if ~isfield(case_opts, 'contrast'), case_opts.contrast = 100; end
    if ~isfield(case_opts, 'background'), case_opts.background = 1; end
    if ~isfield(case_opts, 'forcing'), case_opts.forcing = 'ones'; end
    if ~isfield(case_opts, 'domain'), case_opts.domain = [0, 1, 0, 1]; end
    if ~isfield(case_opts, 'bc_variant'), case_opts.bc_variant = 'dirichlet_all'; end

    xmin = case_opts.domain(1);
    xmax = case_opts.domain(2);
    ymin = case_opts.domain(3);
    ymax = case_opts.domain(4);

    hx = (xmax - xmin) / (nx + 1);
    hy = (ymax - ymin) / (ny + 1);
    x_vec = linspace(xmin + hx, xmax - hx, nx);
    y_vec = linspace(ymin + hy, ymax - hy, ny);
    [X, Y] = meshgrid(x_vec, y_vec);

    low = case_opts.background;
    high = low * case_opts.contrast;

    case_data = struct();
    case_data.bc = default_bc(case_opts.bc_variant);
    case_data.u_exact = [];

    switch lower(case_name)
        case 'constant'
            a = low * ones(ny, nx);
            label = 'Constant Coefficient Diffusion';

        case 'layered_jump'
            if ~isfield(case_opts, 'orientation'), case_opts.orientation = 'vertical'; end
            a = low * ones(ny, nx);
            if strcmpi(case_opts.orientation, 'horizontal')
                a(round(ny / 3):round(2 * ny / 3), :) = high;
                label = sprintf('Layered Jump Diffusion (%gx, horizontal)', case_opts.contrast);
            else
                a(:, round(nx / 3):round(2 * nx / 3)) = high;
                label = sprintf('Layered Jump Diffusion (%gx, vertical)', case_opts.contrast);
            end

        case 'inclusion_jump'
            if ~isfield(case_opts, 'radius'), case_opts.radius = 0.18; end
            if ~isfield(case_opts, 'center'), case_opts.center = [0.5, 0.5]; end
            a = low * ones(ny, nx);
            mask = (X - case_opts.center(1)).^2 + (Y - case_opts.center(2)).^2 <= case_opts.radius^2;
            a(mask) = high;
            label = sprintf('Inclusion Jump Diffusion (%gx)', case_opts.contrast);

        case 'checkerboard_blocks'
            if ~isfield(case_opts, 'blocks_x'), case_opts.blocks_x = 8; end
            if ~isfield(case_opts, 'blocks_y'), case_opts.blocks_y = 8; end
            block_x = min(case_opts.blocks_x, nx);
            block_y = min(case_opts.blocks_y, ny);
            ix = ceil((1:nx) * block_x / nx);
            iy = ceil((1:ny) * block_y / ny);
            a = zeros(ny, nx);
            for j = 1:ny
                for i = 1:nx
                    if mod(ix(i) + iy(j), 2) == 0
                        a(j, i) = low;
                    else
                        a(j, i) = high;
                    end
                end
            end
            label = sprintf('Checkerboard Diffusion (%gx)', case_opts.contrast);

        case 'channel_barrier'
            if ~isfield(case_opts, 'channel_width'), case_opts.channel_width = 0.08; end
            if ~isfield(case_opts, 'barrier_width'), case_opts.barrier_width = 0.05; end
            a = low * ones(ny, nx);
            vertical_channel = abs(X - 0.33) <= case_opts.channel_width / 2;
            horizontal_channel = abs(Y - 0.67) <= case_opts.channel_width / 2;
            vertical_barrier = abs(X - 0.70) <= case_opts.barrier_width / 2;
            horizontal_barrier = abs(Y - 0.38) <= case_opts.barrier_width / 2;
            a(vertical_channel | horizontal_channel) = high;
            a(vertical_barrier | horizontal_barrier) = low / case_opts.contrast;
            label = sprintf('Channel/Barrier Diffusion (%gx)', case_opts.contrast);

        case 'manufactured_smooth'
            a = 1 + 0.25 * X + 0.15 * Y;
            u_exact = sin(pi * X) .* sin(pi * Y);
            ax = 0.25 * ones(size(X));
            ay = 0.15 * ones(size(Y));
            ux = pi * cos(pi * X) .* sin(pi * Y);
            uy = pi * sin(pi * X) .* cos(pi * Y);
            lap_u = -2 * pi^2 * u_exact;
            f = -(ax .* ux + a .* lap_u + ay .* uy);
            case_data.a = a;
            case_data.f = f;
            case_data.u_exact = u_exact;
            case_data.name = 'Manufactured Smooth Diffusion';
            case_data.case_meta = struct('case_name', case_name, 'contrast', 1, 'description', 'smooth manufactured');
            return;

        case 'lognormal'
            % Random log-normal permeability:  kappa = exp(sigma * G)
            % G is a spatially correlated Gaussian field obtained by
            % convolving white noise with a Gaussian kernel of width sigma_px.
            if ~isfield(case_opts, 'sigma_ln'), case_opts.sigma_ln = 2.5;  end
            if ~isfield(case_opts, 'corr_len'), case_opts.corr_len = 0.10; end
            if isfield(case_opts, 'rng_seed'),  rng(case_opts.rng_seed);   end

            % Kernel half-width in grid cells
            sigma_px = max(1, round(case_opts.corr_len / hx));
            half_w   = ceil(3 * sigma_px);
            [kx, ky] = meshgrid(-half_w:half_w, -half_w:half_w);
            kernel   = exp(-(kx.^2 + ky.^2) / (2 * sigma_px^2));
            kernel   = kernel / sum(kernel(:));

            % Standardized correlated Gaussian field
            G = conv2(randn(ny, nx), kernel, 'same');
            G = (G - mean(G(:))) / std(G(:));

            a = exp(case_opts.sigma_ln * G);

            % Record actual contrast for reporting
            case_opts.contrast = round(max(a(:)) / min(a(:)));
            label = sprintf('Log-normal Permeability (\\sigma=%.1f, l_c=%.2f, ~%dx)', ...
                            case_opts.sigma_ln, case_opts.corr_len, case_opts.contrast);

        otherwise
            error('Unknown diffusion benchmark case: %s', case_name);
    end

    case_data.a = a;
    case_data.f = forcing_field(case_opts.forcing, X, Y, nx, ny);
    case_data.name = label;
    case_data.case_meta = struct( ...
        'case_name', case_name, ...
        'contrast', case_opts.contrast, ...
        'description', label ...
    );
end


function bc = default_bc(variant)
    switch lower(variant)
        case 'dirichlet_all'
            bc.left   = struct('type', 'dirichlet', 'value', 0);
            bc.right  = struct('type', 'dirichlet', 'value', 0);
            bc.bottom = struct('type', 'dirichlet', 'value', 0);
            bc.top    = struct('type', 'dirichlet', 'value', 0);

        case 'dirichlet_lr_neumann_tb'
            bc.left   = struct('type', 'dirichlet', 'value', 0);
            bc.right  = struct('type', 'dirichlet', 'value', 0);
            bc.bottom = struct('type', 'neumann',   'value', 0);
            bc.top    = struct('type', 'neumann',   'value', 0);

        otherwise
            error('Unknown BC variant: %s', variant);
    end
end


function f = forcing_field(mode, X, Y, nx, ny)
    switch lower(mode)
        case 'ones'
            f = ones(ny, nx);
        case 'smooth'
            f = 1 + 0.2 * sin(2 * pi * X) .* sin(2 * pi * Y);
        case 'zero'
            f = zeros(ny, nx);
        otherwise
            error('Unknown forcing mode: %s', mode);
    end
end
