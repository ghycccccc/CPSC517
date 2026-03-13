function compare_gmres_preconditioners(K, b)
    % Solver Settings
    tol = 1e-8;
    maxit = 500;
    n = length(b);
    restart = []; % No restart for this analysis
    
    % --- 1. No Preconditioning (Baseline GMRES) ---
    fprintf('Running GMRES (None)...\n');
    [~, ~, ~, ~, res_none] = gmres(K, b, restart, tol, maxit);
    
    % --- 2. Jacobi (Diagonal) ---
    % M = diag(diag(K))
    fprintf('Running GMRES (Jacobi)...\n');
    M_jacobi = diag(diag(K));
    [~, ~, ~, ~, res_jacobi] = gmres(K, b, restart, tol, maxit, M_jacobi);
    
    % --- 3. Incomplete LU (ILU) ---
    % The non-symmetric version of Incomplete Cholesky.
    % 'type','ilutp' is robust for physics matrices.
    fprintf('Running GMRES (ILU)...\n');
    try
        options.type = 'ilutp';
        options.droptol = 1e-3;
        [L_ilu, U_ilu] = ilu(K, options);
        [~, ~, ~, ~, res_ilu] = gmres(K, b, restart, tol, maxit, L_ilu, U_ilu);
    catch
        warning('ILU failed. Matrix may be singular or poorly conditioned.');
        res_ilu = nan;
    end

    % --- 4. Gauss-Seidel (Lower Triangular) ---
    % M = D + L
    fprintf('Running GMRES (Gauss-Seidel)...\n');
    M_gs = tril(K); 
    [~, ~, ~, ~, res_gs] = gmres(K, b, restart, tol, maxit, M_gs);
    
    % --- 5. SOR (Successive Over-Relaxation) ---
    % M = (1/w)*D + L
    fprintf('Running GMRES (SOR)...\n');
    omega = 1.5; % Relaxation factor (tune this between 1 and 2)
    M_sor = tril(K, -1) + (1/omega) * diag(diag(K));
    [~, ~, ~, ~, res_sor] = gmres(K, b, restart, tol, maxit, M_sor);
    
    % --- 6. SSOR (Symmetric Successive Over-Relaxation) ---
    % Uses the triangular parts of K as a cheap approximation of the inverse.
    % fprintf('Running GMRES (SSOR)...\n');
    % D = diag(diag(K));
    % L_tri = tril(K, -1);
    % omega = 1.0; % Relaxation factor
    % M_ssor = (1/omega * D + L_tri) * (diag(1./diag(D))) * (1/omega * D + L_tri');
    % [~, ~, ~, ~, res_ssor] = gmres(K, b, restart, tol, maxit, M_ssor);

    % --- 7. Block Jacobi (2x2 blocks) ---
    % We extract the 2x2 diagonal blocks and invert them
    N = n/2;
    fprintf('Building Block Jacobi...\n');
    block_indices_i = []; block_indices_j = []; block_values = [];
    
    for i = 1:N
        idx = [2*i-1, 2*i]; % Indices for u_i and v_i
        B = full(K(idx, idx)); % Extract 2x2 block
        
        % Store in triplet format for a global sparse M_inv
        [r, c] = meshgrid(idx, idx);
        block_indices_i = [block_indices_i; r(:)];
        block_indices_j = [block_indices_j; c(:)];
        block_values = [block_values; B(:)];
    end
    M_block = sparse(block_indices_i, block_indices_j, block_values, 2*N, 2*N);
    
    % GMRES using the inverse directly as the preconditioner
    [~, ~, ~, ~, res_block] = gmres(K, b, restart, tol, maxit, M_block);


    % --- Visualization ---
    figure('Color', 'w', 'Position', [100, 100, 1000, 500]);
    semilogy(res_none/res_none(1), 'k-', 'LineWidth', 5); hold on;
    semilogy(res_jacobi/res_jacobi(1), 'r-.', 'LineWidth', 2);
    semilogy(res_gs/res_gs(1), 'c-', 'LineWidth', 1.2);
    semilogy(res_sor/res_sor(1), 'b-', 'LineWidth', 1.2);
    % semilogy(res_ssor/res_ssor(1), 'b-.', 'LineWidth', 1.2);
    if ~isnan(res_ilu)
        semilogy(res_ilu/res_ilu(1), 'g-', 'LineWidth', 2);
    end

    if ~isnan(res_block)
        semilogy(res_block/res_block(1), 'y--', 'LineWidth', 1.3);
    end
    
    grid on;
    xlabel('Total Iterations');
    ylabel('Relative Residual Norm');
    title("GMRES Convergence: 2D/3D Navier-Cauchy (K: " + n + "*" + n + ")");
    legend('None', 'Jacobi','Gauss-Seidel','SOR', 'ILU (1e-3)', 'Block', 'Location', 'best');
end