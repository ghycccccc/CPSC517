function results = run_preconditioner_experiment(problem, gmres_opts, amg_opts, pc_opts)
% RUN_PRECONDITIONER_EXPERIMENT
% Compare plain GMRES against several right-preconditioned GMRES variants.
%
% INPUTS:
%   problem.A            - system matrix
%   problem.b            - right-hand side
%   problem.name         - optional display name
%   problem.u_exact      - optional exact solution vector or [ny x nx] grid
%   problem.grid_meta    - optional grid metadata (required for geomg)
%   problem.case_meta    - optional benchmark metadata
%   problem.dof_per_node - DOFs per node (default 1; set 2 for elasticity)
%
%   gmres_opts.tol
%   gmres_opts.max_iter
%   gmres_opts.restart
%
%   amg_opts       - passed to amg_setup/block_amg_setup/preconditioners
%
%   pc_opts.methods     - cell array of method names, default:
%                         {'plain','jacobi','gauss-seidel','sor','ilu','geomg','amg'}
%                         Additional: 'block-jacobi', 'block-amg'
%   pc_opts.sor_omega   - SOR relaxation parameter, default 1.2
%   pc_opts.ilu_setup   - options struct for ilu(), default struct('type','nofill')
%
% OUTPUT:
%   results - struct array with one entry per method
%
% NOTE:
%   Right-preconditioned GMRES:  A * M^{-1} y = b,  x = M^{-1} y.
%   The residual history tracks the true residual norm ||b - A x_k||.

    if ~isfield(problem, 'name'), problem.name = 'Linear System'; end
    A = problem.A;
    b = problem.b;

    dof_per_node = 1;
    if isfield(problem, 'dof_per_node'), dof_per_node = problem.dof_per_node; end
    N_nodes = size(A, 1) / dof_per_node;

    script_dir = fileparts(mfilename('fullpath'));
    fig_dir    = fullfile(script_dir, '..', 'figures');
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
    fig_base   = make_safe_name(problem.name);

    if ~isfield(gmres_opts, 'tol'),      gmres_opts.tol      = 1e-8; end
    if ~isfield(gmres_opts, 'max_iter'), gmres_opts.max_iter = 200;  end
    if ~isfield(gmres_opts, 'restart'),  gmres_opts.restart  = 30;   end

    if ~isfield(amg_opts, 'theta'),            amg_opts.theta            = 0.25; end
    if ~isfield(amg_opts, 'max_levels'),       amg_opts.max_levels       = 8;    end
    if ~isfield(amg_opts, 'coarse_threshold'), amg_opts.coarse_threshold = 40;   end
    if ~isfield(amg_opts, 'nu1'),              amg_opts.nu1              = 1;    end
    if ~isfield(amg_opts, 'nu2'),              amg_opts.nu2              = 1;    end

    if nargin < 4, pc_opts = struct(); end
    if ~isfield(pc_opts, 'methods')
        pc_opts.methods = {'plain', 'jacobi', 'gauss-seidel', 'sor', 'ilu', 'geomg', 'amg'};
    end
    if ~isfield(pc_opts, 'sor_omega'), pc_opts.sor_omega = 1.2; end
    if ~isfield(pc_opts, 'ilu_setup'), pc_opts.ilu_setup = struct('type', 'nofill'); end

    fprintf('=================================================\n');
    fprintf(' Preconditioner Comparison\n');
    fprintf(' %s\n', problem.name);
    fprintf('=================================================\n\n');
    fprintf('System size:   %d x %d\n', size(A, 1), size(A, 2));
    fprintf('Nonzeros:      %d\n', nnz(A));
    fprintf('GMRES restart: %d\n', gmres_opts.restart);
    fprintf('GMRES tol:     %.1e\n\n', gmres_opts.tol);

    if isfield(problem, 'case_meta')
        if isfield(problem.case_meta, 'case_name')
            fprintf('Case:          %s\n', problem.case_meta.case_name);
        end
        if isfield(problem.case_meta, 'contrast')
            fprintf('Contrast:      %g\n', problem.case_meta.contrast);
        end
        if isfield(problem.case_meta, 'description')
            fprintf('Description:   %s\n', problem.case_meta.description);
        end
        fprintf('\n');
    end

    % ------------------------------------------------------------------
    % Build AMG hierarchies before the solve loop
    % ------------------------------------------------------------------
    hierarchy           = [];   amg_setup_time       = NaN;
    geomg_hierarchy     = [];   geomg_setup_time     = NaN;
    block_hierarchy     = [];   block_amg_setup_time = NaN;

    wants_amg       = any(strcmpi(pc_opts.methods, 'amg'));
    wants_geomg     = any(strcmpi(pc_opts.methods, 'geomg'));
    wants_block_amg = any(strcmpi(pc_opts.methods, 'block-amg'));

    if wants_amg
        fprintf('Scalar AMG Setup: building hierarchy...\n');
        t_setup = tic;
        hierarchy = amg_setup(A, amg_opts.theta, ...
                              amg_opts.max_levels, amg_opts.coarse_threshold);
        amg_setup_time = toc(t_setup);
        print_hierarchy_stats(hierarchy, amg_setup_time);
    end

    if wants_geomg
        if ~isfield(problem, 'grid_meta')
            error('problem.grid_meta is required when using the geomg preconditioner.');
        end
        fprintf('Geometric MG Setup: building hierarchy...\n');
        t_setup = tic;
        geomg_hierarchy = geomg_setup(A, problem.grid_meta, ...
                                      amg_opts.max_levels, amg_opts.coarse_threshold);
        geomg_setup_time = toc(t_setup);
        print_hierarchy_stats(geomg_hierarchy, geomg_setup_time);
    end

    if wants_block_amg
        fprintf('Block AMG Setup: building hierarchy...\n');
        t_setup = tic;
        block_hierarchy = block_amg_setup(A, amg_opts.theta, ...
                                          amg_opts.max_levels, amg_opts.coarse_threshold);
        block_amg_setup_time = toc(t_setup);
        print_hierarchy_stats(block_hierarchy, block_amg_setup_time);
    end

    % ------------------------------------------------------------------
    % Hierarchy complexity figure (nnz and DOF count per level)
    % ------------------------------------------------------------------
    if wants_amg || wants_block_amg
        fig_complexity = make_complexity_figure( ...
            hierarchy, block_hierarchy, wants_amg, wants_block_amg);
        save_experiment_figure(fig_complexity, fig_dir, fig_base, 'hierarchy_complexity');
    end

    normb = norm(b);
    if normb == 0, normb = 1; end

    % ------------------------------------------------------------------
    % Problem setup figure: coefficient field (if supplied) + matrix spy
    % ------------------------------------------------------------------
    has_coeff = isfield(problem, 'coeff_field') && ~isempty(problem.coeff_field) ...
                && isfield(problem, 'grid_meta');

    fig_setup = figure('Name', 'Problem Setup', 'Position', [900 120 720 450]);

    if has_coeff
        ax_coeff = subplot(1, 2, 1);
        coeff_label_str = 'Coefficient field';
        if isfield(problem, 'coeff_label')
            coeff_label_str = problem.coeff_label;
        end
        gm = problem.grid_meta;
        imagesc(gm.x_vec, gm.y_vec, problem.coeff_field);
        axis equal tight;
        set(gca, 'YDir', 'normal');
        colorbar;
        colormap(ax_coeff, 'hot');
        xlabel('x');  ylabel('y');
        title(coeff_label_str, 'Interpreter', 'tex');
        ax_mat = subplot(1, 2, 2);
    else
        ax_mat = subplot(1, 1, 1);
    end

    n_A = size(A, 1);
    [si, sj, sv] = find(A);
    max_pts = 40000;
    if numel(si) > max_pts
        pick = round(linspace(1, numel(si), max_pts));
        si = si(pick);  sj = sj(pick);  sv = sv(pick);
    end
    dot_sz = max(1, floor(800 / n_A));
    axes(ax_mat);
    scatter(sj, si, dot_sz, log10(abs(sv) + eps), 'filled');
    set(gca, 'YDir', 'reverse');
    xlim([0.5, n_A + 0.5]);  ylim([0.5, n_A + 0.5]);
    axis square;
    colormap(ax_mat, 'cool');
    cb = colorbar;
    cb.Label.String = 'log_{10}|A_{ij}|';
    xlabel('Column index');  ylabel('Row index');
    title(sprintf('Matrix structure  (n=%d,  nnz=%d)', n_A, nnz(A)));
    drawnow;
    save_experiment_figure(fig_setup, fig_dir, fig_base, 'setup');

    % ------------------------------------------------------------------
    % Solve loop
    % ------------------------------------------------------------------
    results = struct([]);
    fig_conv = figure('Name', 'Preconditioner Convergence', 'Position', [120 120 760 480]);
    set(gca, 'XScale', 'log', 'YScale', 'log');
    hold on;

    for idx = 1:numel(pc_opts.methods)
        method = lower(pc_opts.methods{idx});
        fprintf('--- %s ---\n', upper(method));

        [apply_minv, setup_note, method_setup_time] = build_preconditioner( ...
            method, A, hierarchy, geomg_hierarchy, block_hierarchy, ...
            N_nodes, dof_per_node, amg_opts, pc_opts);

        % Override setup time for pre-built hierarchies
        if strcmp(method, 'amg')
            method_setup_time = amg_setup_time;
        elseif strcmp(method, 'geomg')
            method_setup_time = geomg_setup_time;
        elseif strcmp(method, 'block-amg')
            method_setup_time = block_amg_setup_time;
        end

        if ~isempty(setup_note)
            fprintf('%s', setup_note);
        end

        Afun = @(y) A * apply_minv(y);

        t_solve = tic;
        [y, flag, relres, iter_out, resvec] = gmres( ...
            Afun, b, gmres_opts.restart, gmres_opts.tol, gmres_opts.max_iter);
        solve_time = toc(t_solve);

        x = apply_minv(y);
        iter = flatten_gmres_iter(iter_out, gmres_opts.restart);
        true_resid = norm(b - A * x) / normb;

        results(idx).method     = method;
        results(idx).x          = x;
        results(idx).flag       = flag;
        results(idx).relres     = relres;
        results(idx).iter       = iter;
        results(idx).resvec     = resvec;
        results(idx).true_resid = true_resid;
        results(idx).solve_time = solve_time;
        results(idx).setup_time = method_setup_time;

        fprintf('  Flag:          %d\n', flag);
        fprintf('  Reported resid %.2e\n', relres);
        fprintf('  True residual: %.2e\n', true_resid);
        fprintf('  Iterations:    %d\n', iter);
        if ~isnan(method_setup_time)
            fprintf('  Setup time:    %.3f s\n', method_setup_time);
        end
        fprintf('  Solve time:    %.3f s\n', solve_time);

        if isfield(problem, 'u_exact') && ~isempty(problem.u_exact)
            u_exact = problem.u_exact;
            if ~isvector(u_exact)
                u_exact = reshape(u_exact.', [], 1);
            else
                u_exact = u_exact(:);
            end
            rel_error = norm(x - u_exact) / max(norm(u_exact), eps);
            results(idx).rel_error = rel_error;
            fprintf('  Relative error %.2e\n\n', rel_error);
        else
            results(idx).rel_error = NaN;
            fprintf('\n');
        end

        if isempty(resvec), continue; end

        rel_curve = resvec / resvec(1);
        [line_spec, line_color, line_width] = method_style(method);
        iter_axis = 0 : length(rel_curve) - 1;
        iter_axis(iter_axis == 0) = 1;
        loglog(iter_axis, rel_curve, line_spec, ...
            'Color', line_color, 'LineWidth', line_width, ...
            'DisplayName', method_label(method));
    end

    yline(gmres_opts.tol, 'k:', 'LineWidth', 1.2, 'DisplayName', 'Tolerance');
    grid on;
    xlabel('GMRES Iteration');
    ylabel('Relative Residual Norm');
    title(sprintf('GMRES Convergence: %s', problem.name));
    legend('Location', 'southwest');

    max_len = 1;
    for idx = 1:numel(results)
        max_len = max(max_len, length(results(idx).resvec));
    end
    xlim([1, max(1, max_len - 1)]);
    save_experiment_figure(fig_conv, fig_dir, fig_base, 'convergence');
end


% =========================================================================
function print_hierarchy_stats(h, setup_time)
    fine_rows  = size(h{1}.A, 1);
    fine_nnz   = nnz(h{1}.A);
    total_rows = 0;
    total_nnz  = 0;
    for lev = 1:length(h)
        total_rows = total_rows + size(h{lev}.A, 1);
        total_nnz  = total_nnz  + nnz(h{lev}.A);
    end
    fprintf('  Levels:              %d\n',     length(h));
    fprintf('  Finest  size:        %d DOFs\n', size(h{1}.A,   1));
    fprintf('  Coarsest size:       %d DOFs\n', size(h{end}.A, 1));
    fprintf('  Setup time:          %.3f s\n',  setup_time);
    fprintf('  Grid complexity:     %.3f\n',    total_rows / fine_rows);
    fprintf('  Operator complexity: %.3f\n\n',  total_nnz  / fine_nnz);
end


% =========================================================================
function fig = make_complexity_figure(hierarchy, block_hierarchy, has_amg, has_block_amg)
% Two-panel line-chart figure showing nnz and DOF count per AMG level.
% Each method is a line with markers; data-point values are annotated inline.

    fig = figure('Name', 'AMG Hierarchy Complexity', 'Position', [100 550 960 420]);

    % ---- Collect per-level data -----------------------------------------
    if has_amg && ~isempty(hierarchy)
        L_s      = length(hierarchy);
        amg_nnz  = arrayfun(@(l) nnz(hierarchy{l}.A),     1:L_s);
        amg_dofs = arrayfun(@(l) size(hierarchy{l}.A, 1), 1:L_s);
        amg_oc   = sum(amg_nnz)  / amg_nnz(1);
        amg_gc   = sum(amg_dofs) / amg_dofs(1);
    else
        L_s = 0;
    end

    if has_block_amg && ~isempty(block_hierarchy)
        L_b       = length(block_hierarchy);
        bamg_nnz  = arrayfun(@(l) nnz(block_hierarchy{l}.A),     1:L_b);
        bamg_dofs = arrayfun(@(l) size(block_hierarchy{l}.A, 1), 1:L_b);
        bamg_oc   = sum(bamg_nnz)  / bamg_nnz(1);
        bamg_gc   = sum(bamg_dofs) / bamg_dofs(1);
    else
        L_b = 0;
    end

    L_max = max(L_s, L_b);

    clr_amg  = [0.00, 0.25, 0.90];   % blue — scalar AMG
    clr_bamg = [0.00, 0.65, 0.60];   % teal — block AMG

    % ---- Left panel: nnz per level --------------------------------------
    subplot(1, 2, 1);
    hold on;

    leg_entries = {};
    if L_s > 0
        semilogy(1:L_s, amg_nnz, '-o', ...
            'Color', clr_amg, 'LineWidth', 1.8, ...
            'MarkerSize', 7, 'MarkerFaceColor', clr_amg);
        annotate_line_points(1:L_s, amg_nnz, clr_amg, 'above');
        leg_entries{end+1} = sprintf('Scalar AMG  (OC = %.2f\\times)', amg_oc);
    end
    if L_b > 0
        semilogy(1:L_b, bamg_nnz, '-s', ...
            'Color', clr_bamg, 'LineWidth', 1.8, ...
            'MarkerSize', 7, 'MarkerFaceColor', clr_bamg);
        annotate_line_points(1:L_b, bamg_nnz, clr_bamg, 'below');
        leg_entries{end+1} = sprintf('Block AMG  (OC = %.2f\\times)', bamg_oc);
    end

    set(gca, 'YScale', 'log');
    xlabel('Level');
    ylabel('nnz(A_l)');
    title('nnz per hierarchy level', 'FontSize', 10);
    xticks(1:L_max);  xlim([0.5, L_max + 0.5]);
    grid on;
    if ~isempty(leg_entries)
        legend(leg_entries, 'Location', 'southwest', 'FontSize', 9);
    end

    % ---- Right panel: DOF count per level --------------------------------
    subplot(1, 2, 2);
    hold on;

    leg_entries = {};
    if L_s > 0
        semilogy(1:L_s, amg_dofs, '-o', ...
            'Color', clr_amg, 'LineWidth', 1.8, ...
            'MarkerSize', 7, 'MarkerFaceColor', clr_amg);
        annotate_line_points(1:L_s, amg_dofs, clr_amg, 'above');
        leg_entries{end+1} = sprintf('Scalar AMG  (GC = %.2f\\times)', amg_gc);
    end
    if L_b > 0
        semilogy(1:L_b, bamg_dofs, '-s', ...
            'Color', clr_bamg, 'LineWidth', 1.8, ...
            'MarkerSize', 7, 'MarkerFaceColor', clr_bamg);
        annotate_line_points(1:L_b, bamg_dofs, clr_bamg, 'below');
        leg_entries{end+1} = sprintf('Block AMG  (GC = %.2f\\times)', bamg_gc);
    end

    set(gca, 'YScale', 'log');
    xlabel('Level');
    ylabel('DOFs at level l');
    title('DOF count per hierarchy level', 'FontSize', 10);
    xticks(1:L_max);  xlim([0.5, L_max + 0.5]);
    grid on;
    if ~isempty(leg_entries)
        legend(leg_entries, 'Location', 'southwest', 'FontSize', 9);
    end

    sgtitle('AMG Hierarchy Complexity', 'FontSize', 12, 'FontWeight', 'bold');
end


function annotate_line_points(xs, ys, clr, side)
% Place a numeric label beside each data point on a log-y line chart.
% side = 'above' places label above the point; 'below' places it below.
% A small fixed multiplicative offset keeps labels clear of the marker.
    if strcmpi(side, 'above')
        va = 'bottom';  yfac = 1.25;
    else
        va = 'top';     yfac = 0.80;
    end
    for k = 1:numel(xs)
        v = ys(k);
        if v > 0
            lbl = format_nnz(v);
            text(xs(k), v * yfac, lbl, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', va, ...
                'FontSize', 7.5, 'Color', clr * 0.75);
        end
    end
end


function s = format_nnz(v)
% Compact label: use k/M suffix for large values to avoid overcrowding.
    if v >= 1e6
        s = sprintf('%.1fM', v / 1e6);
    elseif v >= 1e3
        s = sprintf('%.1fk', v / 1e3);
    else
        s = sprintf('%d', round(v));
    end
end


% =========================================================================
function [apply_minv, setup_note, setup_time] = build_preconditioner( ...
        method, A, hierarchy, geomg_hierarchy, block_hierarchy, ...
        N_nodes, dof_per_node, amg_opts, pc_opts)

    n = size(A, 1);
    d = diag(A);
    setup_note = '';
    setup_time = NaN;

    switch method
        case 'plain'
            apply_minv = @(r) r;
            setup_time = 0;

        case 'jacobi'
            if any(abs(d) < 1e-14)
                error('Jacobi: zero diagonal entry detected.');
            end
            apply_minv = @(r) r ./ d;
            setup_time = 0;

        case 'block-jacobi'
            t_bj = tic;
            D_inv_sp = build_block_diag_inv(A, N_nodes, dof_per_node);
            bj_time  = toc(t_bj);
            setup_note = sprintf('  Block-Jacobi build:  %.3f s  (block size %d)\n', ...
                                 bj_time, dof_per_node);
            apply_minv = @(r) D_inv_sp * r;
            setup_time = bj_time;

        case 'gauss-seidel'
            M = tril(A);
            apply_minv = @(r) M \ r;
            setup_time = 0;

        case 'sor'
            omega = pc_opts.sor_omega;
            if omega <= 0 || omega >= 2
                error('SOR omega must satisfy 0 < omega < 2.');
            end
            M = tril(A, -1) + spdiags(d / omega, 0, n, n);
            setup_note = sprintf('  omega:               %.2f\n', omega);
            apply_minv = @(r) M \ r;
            setup_time = 0;

        case 'ilu'
            t_ilu = tic;
            [L, U] = ilu(A, pc_opts.ilu_setup);
            ilu_time = toc(t_ilu);
            setup_note = sprintf('  ILU setup:           %.3f s\n', ilu_time);
            apply_minv = @(r) U \ (L \ r);
            setup_time = ilu_time;

        case 'geomg'
            if isempty(geomg_hierarchy)
                error('Geometric MG hierarchy is not available.');
            end
            setup_note = sprintf('  Geomg levels:        %d\n', length(geomg_hierarchy));
            apply_minv = @(r) geomg_preconditioner(geomg_hierarchy, r, amg_opts.nu1, amg_opts.nu2);
            setup_time = 0;

        case 'amg'
            if isempty(hierarchy)
                error('Scalar AMG hierarchy is not available.');
            end
            nu1 = amg_opts.nu1;
            nu2 = amg_opts.nu2;
            apply_minv = @(r) amg_preconditioner(hierarchy, r, nu1, nu2);
            setup_time = 0;

        case 'block-amg'
            if isempty(block_hierarchy)
                error('Block AMG hierarchy is not available.');
            end
            nu1 = amg_opts.nu1;
            nu2 = amg_opts.nu2;
            apply_minv = @(r) block_amg_vcycle(block_hierarchy, 1, r, zeros(size(r)), nu1, nu2);
            setup_time = 0;

        otherwise
            error('Unknown preconditioner method: %s', method);
    end
end


% =========================================================================
function D_inv = build_block_diag_inv(A, N_nodes, dof_blk)
% Sparse block-diagonal inverse: (dof_blk x dof_blk) diagonal blocks.
% Assumes interleaved DOF ordering: node i owns DOFs (i-1)*dof_blk+1 : i*dof_blk.

    n   = size(A, 1);
    nel = dof_blk^2 * N_nodes;
    II  = zeros(nel, 1);
    JJ  = zeros(nel, 1);
    VV  = zeros(nel, 1);
    ptr = 0;

    for i = 1:N_nodes
        ri    = (i - 1) * dof_blk + (1:dof_blk);
        D_blk = full(A(ri, ri));

        if dof_blk == 2
            det_D = D_blk(1,1)*D_blk(2,2) - D_blk(1,2)*D_blk(2,1);
            if abs(det_D) < 1e-14
                d_avg = max((abs(D_blk(1,1)) + abs(D_blk(2,2))) / 2, 1e-14);
                D_inv_blk = eye(2) / d_avg;
            else
                D_inv_blk = [D_blk(2,2), -D_blk(1,2); -D_blk(2,1), D_blk(1,1)] / det_D;
            end
        else
            D_inv_blk = inv(D_blk);
        end

        for r = 1:dof_blk
            for c = 1:dof_blk
                ptr     = ptr + 1;
                II(ptr) = ri(r);
                JJ(ptr) = ri(c);
                VV(ptr) = D_inv_blk(r, c);
            end
        end
    end

    D_inv = sparse(II(1:ptr), JJ(1:ptr), VV(1:ptr), n, n);
end


% =========================================================================
function iter = flatten_gmres_iter(iter_out, restart)
    if numel(iter_out) == 2
        outer = iter_out(1);
        inner = iter_out(2);
        if inner == 0
            iter = (outer - 1) * restart;
        else
            iter = (outer - 1) * restart + inner;
        end
    else
        iter = iter_out;
    end
end


% =========================================================================
function label = method_label(method)
    switch method
        case 'plain',         label = 'Plain GMRES';
        case 'jacobi',        label = 'Jacobi';
        case 'block-jacobi',  label = 'Block Jacobi (2\times2)';
        case 'gauss-seidel',  label = 'Gauss-Seidel';
        case 'sor',           label = 'SOR';
        case 'ilu',           label = 'ILU(0)';
        case 'geomg',         label = 'Geometric MG';
        case 'amg',           label = 'Scalar AMG';
        case 'block-amg',     label = 'Block AMG';
        otherwise,            label = method;
    end
end


% =========================================================================
function [line_spec, line_color, line_width] = method_style(method)
    switch method
        case 'plain'
            line_spec = '--';  line_color = [0.00, 0.00, 0.00];  line_width = 2.8;
        case 'jacobi'
            line_spec = '-';   line_color = [0.00, 0.75, 0.75];  line_width = 1.8;
        case 'block-jacobi'
            line_spec = '-';   line_color = [0.55, 0.00, 0.75];  line_width = 1.8;
        case 'gauss-seidel'
            line_spec = '-';   line_color = [0.00, 0.50, 0.00];  line_width = 1.8;
        case 'sor'
            line_spec = '-';   line_color = [0.75, 0.00, 0.75];  line_width = 1.8;
        case 'ilu'
            line_spec = '-';   line_color = [0.85, 0.10, 0.10];  line_width = 1.8;
        case 'geomg'
            line_spec = '-';   line_color = [0.85, 0.45, 0.10];  line_width = 1.8;
        case 'amg'
            line_spec = '-';   line_color = [0.00, 0.25, 0.90];  line_width = 1.8;
        case 'block-amg'
            line_spec = '-';   line_color = [0.00, 0.65, 0.60];  line_width = 2.2;
        otherwise
            line_spec = '-';   line_color = [0.20, 0.20, 0.20];  line_width = 1.8;
    end
end


% =========================================================================
function safe = make_safe_name(str)
    safe = lower(strtrim(str));
    safe = regexprep(safe, '[^a-z0-9]+', '_');
    safe = regexprep(safe, '^_+|_+$', '');
    if isempty(safe), safe = 'experiment'; end
    if length(safe) > 60, safe = safe(1:60); end
end


% =========================================================================
function save_experiment_figure(fig, fig_dir, base_name, fig_type)
    filename = fullfile(fig_dir, sprintf('%s_%s.png', base_name, fig_type));
    print(fig, filename, '-dpng', '-r150');
    fprintf('  [saved] %s\n', filename);
end
