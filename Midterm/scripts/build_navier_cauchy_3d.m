function [K, b] = build_navier_cauchy_3d(nx, ny, nz, h, lambda, mu, fx, fy, fz)
    N = nx * ny * nz;
    total_dofs = 3 * N;
    
    % Constants
    C1 = (lambda + 2*mu) / h^2;
    C2 = mu / h^2;
    C3 = (lambda + mu) / (4 * h^2);
    
    % Triplets for efficiency
    I = []; J = []; V = [];
    b = zeros(total_dofs, 1);
    
    % Indexing Helpers
    getNode = @(i, j, k) (k-1)*nx*ny + (j-1)*nx + i;
    getDofs = @(nIdx) [3*nIdx-2; 3*nIdx-1; 3*nIdx]; % Returns [u_dof, v_dof, w_dof]

    for k = 1:nz
        for j = 1:ny
            for i = 1:nx
                currNode = getNode(i, j, k);
                dofs = getDofs(currNode); % [ux, uy, uz]
                
                % Load Vector
                b(dofs) = [fx(i,j,k); fy(i,j,k); fz(i,j,k)];
                
                % 27-point Stencil (-1 to +1 in all directions)
                for dk = -1:1, for dj = -1:1, for di = -1:1
                    ni = i+di; nj = j+dj; nk = k+dk;
                    if ni>0 && ni<=nx && nj>0 && nj<=ny && nk>0 && nk<=nz
                        neighborNode = getNode(ni, nj, nk);
                        nDofs = getDofs(neighborNode);
                        
                        % --- 1. Laplacian & Normal Gradients (Main Diagonals) ---
                        if di==0 && dj==0 && dk==0 % Self
                            vals = -2*(C1 + 2*C2) * ones(3,1);
                            I = [I, dofs']; J = [J, dofs']; V = [V, vals'];
                        elseif abs(di)+abs(dj)+abs(dk) == 1 % Orthogonal Neighbors
                            if di~=0, v=[C1, C2, C2]; end
                            if dj~=0, v=[C2, C1, C2]; end
                            if dk~=0, v=[C2, C2, C1]; end
                            I = [I, dofs']; J = [J, nDofs']; V = [V, v];
                        end
                        
                        % --- 2. Mixed Derivatives (Cross Coupling) ---
                        % Coupling u-v via d2/dxdy
                        if di~=0 && dj~=0 && dk==0
                            val = di*dj*C3;
                            I = [I, dofs(1), dofs(2)]; J = [J, nDofs(2), nDofs(1)]; V = [V, val, val];
                        end
                        % Coupling u-w via d2/dxdz
                        if di~=0 && dj==0 && dk~=0
                            val = di*dk*C3;
                            I = [I, dofs(1), dofs(3)]; J = [J, nDofs(3), nDofs(1)]; V = [V, val, val];
                        end
                        % Coupling v-w via d2/dydz
                        if di==0 && dj~=0 && dk~=0
                            val = dj*dk*C3;
                            I = [I, dofs(2), dofs(3)]; J = [J, nDofs(3), nDofs(2)]; V = [V, val, val];
                        end
                    end
                end, end, end
            end
        end
    end
    K = sparse(I, J, V, total_dofs, total_dofs);
    
    % --- SYMMETRIC DIRICHLET BC (Fixed wall at i=1) ---
    fixed_nodes = [];
    for k=1:nz, for j=1:ny, fixed_nodes = [fixed_nodes, getNode(1,j,k)]; end, end
    fixed_dofs = reshape(getDofs(fixed_nodes), [], 1);
    
    % Symmetric zeroing: b = b - K(:,fixed)*x_fixed (here x_fixed=0)
    K(fixed_dofs, :) = 0;
    K(:, fixed_dofs) = 0;
    for d = 1:length(fixed_dofs)
        idx = fixed_dofs(d);
        K(idx, idx) = 1;
        b(idx) = 0;
    end
end