function [K, b] = build_navier_cauchy(nx, ny, h, lambda, mu, fx, fy, ordering)
    % ordering: 'node' for [u1, v1, u2, v2...] or 'component' for [u1..uN, v1..vN]
    
    N = nx * ny;
    total_dofs = 2 * N;
    
    % Pre-calculate constants
    C1 = (lambda + 2*mu) / h^2;
    C2 = mu / h^2;
    C3 = (lambda + mu) / (4 * h^2);
    
    I = []; J = []; V = [];
    b = zeros(total_dofs, 1);
    
    % --- Indexing Logic ---
    getNode = @(i, j) (j-1)*nx + i;
    
    if strcmpi(ordering, 'node')
        getUdof = @(nIdx) 2*nIdx - 1;
        getVdof = @(nIdx) 2*nIdx;
    else % component-wise
        getUdof = @(nIdx) nIdx;
        getVdof = @(nIdx) nIdx + N;
    end

    for j = 1:ny
        for i = 1:nx
            currNode = getNode(i, j);
            u_row = getUdof(currNode);
            v_row = getVdof(currNode);
            
            b(u_row) = fx(i,j);
            b(v_row) = fy(i,j);
            
            for dj = -1:1
                for di = -1:1
                    ni = i + di; nj = j + dj;
                    if ni > 0 && ni <= nx && nj > 0 && nj <= ny
                        neighborNode = getNode(ni, nj);
                        u_col = getUdof(neighborNode);
                        v_col = getVdof(neighborNode);
                        
                        % 1. Laplacian terms
                        if di == 0 && dj == 0 % Self
                            I = [I, u_row, v_row]; J = [J, u_row, v_row];
                            V = [V, -2*(C1 + C2), -2*(C1 + C2)];
                        elseif di ~= 0 && dj == 0 % Horizontal
                            I = [I, u_row, v_row]; J = [J, u_col, v_col];
                            V = [V, C1, C2];
                        elseif di == 0 && dj ~= 0 % Vertical
                            I = [I, u_row, v_row]; J = [J, u_col, v_col];
                            V = [V, C2, C1];
                        end
                        
                        % 2. Mixed Derivative (Coupling)
                        if di ~= 0 && dj ~= 0
                            val = di * dj * C3;
                            I = [I, u_row, v_row]; J = [J, v_col, u_col];
                            V = [V, val, val];
                        end
                    end
                end
            end
        end
    end
    
    K = sparse(I, J, V, total_dofs, total_dofs);
    
    % --- Apply Dirichlet BC (Fixed left wall) ---
    for j = 1:ny
        fNode = getNode(1, j);
        u_dof = getUdof(fNode); v_dof = getVdof(fNode);
        K(u_dof, :) = 0; K(u_dof, u_dof) = 1; b(u_dof) = 0;
        K(v_dof, :) = 0; K(v_dof, v_dof) = 1; b(v_dof) = 0;
    end
end