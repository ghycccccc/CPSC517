function [res_node, res_comp] =  compare_preconditioners(K_n, b_n, K_c, b_c)
    % Settings for the iterative solver
    tol = 1e-8;
    maxit = 1000;
    n = length(b_n);
    
    % --- 1. No Preconditioning (Baseline CG) ---
    fprintf('Running GMRES for Node-based...\n');
    [~, ~, ~, ~, res_node] = gmres(K_n, b_n, [], tol, maxit);


    fprintf('Running GMRES for Component-based...\n');
    [~, ~, ~, ~, res_comp] = gmres(K_c, b_c, [], tol, maxit);

    % --- Plotting the Results ---
    figure('Color', 'w', 'Name', 'Mode Convergence Comparison');
    semilogy(res_node, 'k-', 'LineWidth', 1.5); hold on;
    semilogy(res_comp, 'r--', 'LineWidth', 1.5);
    %semilogy(res_ssor, 'b-.', 'LineWidth', 1.5);
    % if ~isnan(res_ic)
    %     semilogy(res_ic, 'g-', 'LineWidth', 2);
    % end
    
    grid on;
    xlabel('Iteration Number');
    ylabel('Relative Residual Norm');
    title('Convergence of PCG for 2D Navier-Cauchy');
    legend('Node', 'Component');
end