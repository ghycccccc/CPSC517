function [K, b] = dirichlet_demo_case(nx, ny, h, mu, lambda)
    [K, b] = build_navier_cauchy(nx, ny, h, mu, lambda, rand(nx, ny), rand(nx, ny), "node");
    
    % --- Step 1: Identify boundary nodes in the nx by ny grid ---
    % Nodes are numbered 1 to nx along the x-axis, then moving up the y-axis.
    % Node index = (j-1)*nx + i
    
    boundary_nodes = [];
    
    % Bottom edge (j=1)
    boundary_nodes = [boundary_nodes, 1:nx];
    % Top edge (j=ny)
    boundary_nodes = [boundary_nodes, (ny-1)*nx + (1:nx)];
    % Left edge (i=1), excluding corners already counted
    left_nodes = ((2:ny-1) - 1)*nx + 1;
    boundary_nodes = [boundary_nodes, left_nodes];
    % Right edge (i=nx), excluding corners already counted
    right_nodes = ((2:ny-1) - 1)*nx + nx;
    boundary_nodes = [boundary_nodes, right_nodes];
    
    % --- Step 2: Map nodes to Degrees of Freedom (DOFs) ---
    % For node-wise ordering, node 'n' has u at 2n-1, and v at 2n.
    boundary_dofs = zeros(1, 2*length(boundary_nodes));
    boundary_dofs(1:2:end) = 2 * boundary_nodes - 1; % u-components
    boundary_dofs(2:2:end) = 2 * boundary_nodes;     % v-components
    
    % --- Step 3: Apply the Symmetric Dirichlet Condition ---
    % Zero out the rows and columns for all boundary DOFs
    K(boundary_dofs, :) = 0;
    K(:, boundary_dofs) = 0;
    
    % Set the diagonal entries to 1 to preserve matrix rank
    % (A loop is used here for clarity, but for massive matrices, 
    % spdiags or linear indexing is faster).
    for k = boundary_dofs
        K(k, k) = 1;
        b(k) = 0;
    end
    
    compare_gmres_preconditioners(K, b);
end