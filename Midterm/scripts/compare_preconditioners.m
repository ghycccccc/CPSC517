function [res_none, res_jacobi] =  compare_preconditioners(K, b)
    % Settings for the iterative solver
    tol = 1e-8;
    maxit = 1000;
    n = length(b);
    
    % --- 1. No Preconditioning (Baseline CG) ---
    fprintf('Running CG (No Preconditioner)...\n');
    %[~, ~, ~, ~, res_none] = pcg(K, b, tol, maxit);
    % GMRES
    [~, ~, ~, ~, res_none] = gmres(K, b, [], tol, maxit);
    
    % --- 2. Jacobi (Diagonal) Preconditioning ---
    % Simplest: M = diag(diag(K))
    fprintf('Running PCG (Jacobi)...\n');
    M_jacobi = diag(diag(K));
    [~, ~, ~, ~, res_jacobi] = pcg(K, b, tol, maxit, M_jacobi);

    
    % % --- 3. Incomplete Cholesky (IC) ---
    % % This is the "standard" for SPD physics problems. 
    % % We use a drop tolerance to control sparsity.
    % fprintf('Running PCG (Incomplete Cholesky)...\n');
    % try
    %     % 'ict' type is Incomplete Cholesky with Threshold
    %     % droptol: entries smaller than this are discarded
    %     L_ic = ichol(K, struct('type','ict','droptol',1e-3));
    %     [~, ~, ~, ~, res_ic] = pcg(K, b, tol, maxit, L_ic, L_ic');
    % catch
    %     warning('ichol failed. Matrix might be poorly conditioned for IC.');
    %     res_ic = nan;
    % end
    
    % % --- 4. Successive Over-Relaxation (SOR) ---
    % % Often used as a smoother in multigrid. 
    % % Approximated here via the lower triangular part of K.
    % fprintf('Running PCG (SOR-like / SSOR)...\n');
    % L_tri = tril(K);
    % D_vec = diag(K);
    % M_ssor = L_tri * diag(1./D_vec) * L_tri';
    % [~, ~, ~, ~, res_ssor] = pcg(K, b, tol, maxit, M_ssor);

    % --- Plotting the Results ---
    figure('Color', 'w', 'Name', 'Solver Convergence Comparison');
    semilogy(res_none, 'k-', 'LineWidth', 1.5); hold on;
    disp(res_none);
    semilogy(res_jacobi, 'r--', 'LineWidth', 1.5);
    disp(res_jacobi);
    %semilogy(res_ssor, 'b-.', 'LineWidth', 1.5);
    % if ~isnan(res_ic)
    %     semilogy(res_ic, 'g-', 'LineWidth', 2);
    % end
    
    grid on;
    xlabel('Iteration Number');
    ylabel('Relative Residual Norm');
    title('Convergence of PCG for 2D Navier-Cauchy');
    legend('None (CG)', 'Jacobi', 'SSOR', 'Incomplete Cholesky (1e-3)', ...
           'Location', 'best');
end