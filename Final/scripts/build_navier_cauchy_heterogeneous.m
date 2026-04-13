function [K, b] = build_navier_cauchy_heterogeneous(nx, ny, h, lambda, mu, fx, fy)
% BUILD_NAVIER_CAUCHY_HETEROGENEOUS
%   Assembles the stiffness matrix K and load vector b for the 2D
%   heterogeneous Navier-Cauchy (linear elasticity) problem using
%   second-order central finite differences.
%
%   The governing equations (strong form, heterogeneous) are:
%
%     d/dx[(l+2m) du/dx + l dv/dy] + d/dy[m(du/dy + dv/dx)] + fx = 0
%     d/dx[m(du/dy + dv/dx)] + d/dy[l du/dx + (l+2m) dv/dy] + fy = 0
%
%   Expanding via product rule introduces first-order "gradient correction"
%   terms absent in the homogeneous case. Full derivation in project report.
%
%   DOF ordering: node-based, [u1, v1, u2, v2, ..., uN, vN]
%   Boundary condition: Dirichlet zero on all four walls.
%
% INPUTS:
%   nx, ny    - number of grid nodes in x and y directions
%   h         - uniform grid spacing (scalar)
%   lambda    - Lame parameter lambda, size [ny x nx]
%   mu        - Lame parameter mu,     size [ny x nx]
%   fx, fy    - body force arrays,     size [ny x nx]
%               (following Matlab matrix convention: row=y, col=x)
%
% OUTPUTS:
%   K  - sparse stiffness matrix, size [2*nx*ny x 2*nx*ny]
%   b  - load vector,             size [2*nx*ny x 1]
%
% NOTE ON SIGN CONVENTION:
%   Following the derivation, the assembled equation at each interior node
%   is written as K*x = b where b contains the negated body forces
%   (moved from LHS to RHS). The diagonal of K is positive; off-diagonal
%   neighbor entries are negative (plus gradient corrections).

    % ------------------------------------------------------------------
    % 0. Validate inputs
    % ------------------------------------------------------------------
    assert(all(size(lambda) == [ny, nx]), ...
        'lambda must be [ny x nx] to match Matlab row=y, col=x convention');
    assert(all(size(mu)     == [ny, nx]), ...
        'mu must be [ny x nx]');
    assert(all(size(fx)     == [ny, nx]), 'fx must be [ny x nx]');
    assert(all(size(fy)     == [ny, nx]), 'fy must be [ny x nx]');

    % ------------------------------------------------------------------
    % 1. Indexing helpers  (node-based DOF ordering)
    % ------------------------------------------------------------------
    N          = nx * ny;
    total_dofs = 2 * N;

    % Node index from grid coordinates (col-major: i=x-index, j=y-index)
    getNode = @(i, j) (j - 1) * nx + i;

    % DOF indices for u and v at a given node index
    getUdof = @(nIdx) 2 * nIdx - 1;
    getVdof = @(nIdx) 2 * nIdx;

    % ------------------------------------------------------------------
    % 2. Pre-compute material gradients at every node via central diff
    %    (one-sided differences at boundaries -- but those nodes will
    %     have BCs applied, so accuracy there is irrelevant)
    % ------------------------------------------------------------------
    % Allocate gradient arrays  [ny x nx],  same layout as lambda/mu
    dlambda_dx = zeros(ny, nx);   % d(lambda)/dx
    dlambda_dy = zeros(ny, nx);   % d(lambda)/dy
    dmu_dx     = zeros(ny, nx);   % d(mu)/dx
    dmu_dy     = zeros(ny, nx);   % d(mu)/dy

    % Interior nodes: central differences
    % Matlab convention: lambda(j, i) -> row j = y-direction, col i = x-direction
    dlambda_dx(:, 2:nx-1) = (lambda(:, 3:nx  ) - lambda(:, 1:nx-2)) / (2*h);
    dlambda_dy(2:ny-1, :) = (lambda(3:ny,   :) - lambda(1:ny-2, :)) / (2*h);
    dmu_dx    (:, 2:nx-1) = (mu    (:, 3:nx  ) - mu    (:, 1:nx-2)) / (2*h);
    dmu_dy    (2:ny-1, :) = (mu    (3:ny,   :) - mu    (1:ny-2, :)) / (2*h);

    % Boundary nodes: one-sided (forward/backward) differences
    dlambda_dx(:, 1 ) = (lambda(:, 2   ) - lambda(:, 1   )) / h;
    dlambda_dx(:, nx) = (lambda(:, nx  ) - lambda(:, nx-1)) / h;
    dlambda_dy(1,  :) = (lambda(2,   : ) - lambda(1,   : )) / h;
    dlambda_dy(ny, :) = (lambda(ny,  : ) - lambda(ny-1, :)) / h;

    dmu_dx(:, 1 ) = (mu(:, 2   ) - mu(:, 1   )) / h;
    dmu_dx(:, nx) = (mu(:, nx  ) - mu(:, nx-1)) / h;
    dmu_dy(1,  :) = (mu(2,   : ) - mu(1,   : )) / h;
    dmu_dy(ny, :) = (mu(ny,  : ) - mu(ny-1, :)) / h;

    % ------------------------------------------------------------------
    % 3. Assemble K via triplet lists  (I, J, V)
    % ------------------------------------------------------------------
    % Each interior node contributes at most 9 neighbor pairs x 2 DOFs
    % x 2 equations = 36 entries. Over-allocate; sparse() sums duplicates.
    est_nnz = 36 * N;
    I = zeros(est_nnz, 1);
    J = zeros(est_nnz, 1);
    V = zeros(est_nnz, 1);
    ptr = 0;   % points to last filled position

    b = zeros(total_dofs, 1);

    % Helper: append a single triplet entry
    function append(row, col, val)
        ptr       = ptr + 1;
        I(ptr)    = row;
        J(ptr)    = col;
        V(ptr)    = val;
    end

    for j = 1:ny          % j = y-grid index
        for i = 1:nx      % i = x-grid index

            % ---- Local material values at (i,j) ----------------------
            lam  = lambda(j, i);
            muu  = mu    (j, i);

            % Node-local second-order coefficients (from midterm, now local)
            C1_loc = (lam + 2*muu) / h^2;
            C2_loc = muu           / h^2;
            C3_loc = (lam +  muu ) / (4 * h^2);

            % Node-local gradient correction terms.
            % These enter the first-order derivative terms after product rule.
            %
            % From x-momentum derivation:
            %   coeff of du/dx  gets  d(lam+2mu)/dx / (2h) * (1/(2h)) * h^2
            %   which simplifies to   d(lam+2mu)/dx / (4h) ... but let us
            %   keep the factors explicit for clarity.
            %
            % For axis-aligned neighbors at distance h:
            %   first-order central diff of f at node i gives
            %   (f_{i+1} - f_{i-1})/(2h), contributing ±1/(2h) to each neighbor.
            %   Combined with the 1/(2h) from du/dx, the gradient correction
            %   coefficient for u_{i+1,j} in the x-momentum equation is:
            %       +d(lam+2mu)/dx * 1/(2h) * 1/(2h) * h^2
            %     = +d(lam+2mu)/dx / 4
            %   and for u_{i-1,j}:
            %       -d(lam+2mu)/dx / 4
            %
            % Gradient corrections for axis-aligned neighbors:
            Gx_lam2mu = (dlambda_dx(j,i) + 2*dmu_dx(j,i)) / 4;  % d(lam+2mu)/dx /4
            Gy_lam2mu = (dlambda_dy(j,i) + 2*dmu_dy(j,i)) / 4;  % d(lam+2mu)/dy /4
            Gx_mu     = dmu_dx(j,i) / 4;   % d(mu)/dx / 4
            Gy_mu     = dmu_dy(j,i) / 4;   % d(mu)/dy / 4

            % Gradient corrections for diagonal neighbors (cross-derivative terms).
            % From x-momentum: coeff of v_{i+1,j+1} acquires corrections from
            %   dl/dx * dv/dy  and  dmu/dy * dv/dx  first-order terms:
            %       dl/dx * 1/(2h) * 1/(2h) = dl/dx / (4h^2) ... times h^2 -> /4
            %   Similarly for dmu/dy term.
            Gx_lam  = dlambda_dx(j,i) / 4;   % for cross terms involving dl/dx
            Gy_lam  = dlambda_dy(j,i) / 4;   % for cross terms involving dl/dy
            Gx_muC  = dmu_dx(j,i)     / 4;   % for cross terms involving dmu/dx
            Gy_muC  = dmu_dy(j,i)     / 4;   % for cross terms involving dmu/dy

            % ---- DOF rows for this node ---------------------------------
            currNode = getNode(i, j);
            u_row    = getUdof(currNode);
            v_row    = getVdof(currNode);

            % ---- RHS load vector ----------------------------------------
            b(u_row) = fx(j, i);
            b(v_row) = fy(j, i);

            % ==============================================================
            % STENCIL ASSEMBLY
            % Loop over the 3x3 neighborhood (di, dj) in {-1,0,1}^2
            % ==============================================================
            for dj = -1:1
                for di = -1:1
                    ni = i + di;
                    nj = j + dj;

                    % Skip out-of-bound neighbors (Dirichlet BC enforced later)
                    if ni < 1 || ni > nx || nj < 1 || nj > ny
                        continue;
                    end

                    neighborNode = getNode(ni, nj);
                    u_col        = getUdof(neighborNode);
                    v_col        = getVdof(neighborNode);

                    % --------------------------------------------------
                    % CASE 1: Self-interaction  (di=0, dj=0)
                    % --------------------------------------------------
                    if di == 0 && dj == 0
                        % Diagonal entry: positive, from second-order terms only.
                        % No first-order self-coupling (central diff has zero weight
                        % at center for first-order derivatives).
                        append(u_row, u_col,  2*(C1_loc + C2_loc));
                        append(v_row, v_col,  2*(C1_loc + C2_loc));

                    % --------------------------------------------------
                    % CASE 2: Horizontal neighbor  (di=±1, dj=0)
                    % --------------------------------------------------
                    elseif dj == 0 && di ~= 0
                        % X-momentum / u equation:
                        %   Second-order: -C1_loc (from d^2u/dx^2)
                        %   Gradient correction: +sign(di) * Gx_lam2mu
                        %     (d(lam+2mu)/dx * du/dx term; di gives the sign)
                        append(u_row, u_col, -C1_loc + di * Gx_lam2mu);

                        % Y-momentum / v equation:
                        %   Second-order: -C2_loc (from d^2v/dx^2, mu coefficient)
                        %   Gradient correction: +sign(di) * Gx_mu
                        %     (dmu/dx * dv/dx term)
                        append(v_row, v_col, -C2_loc + di * Gx_mu);

                        % No u-v or v-u coupling for axis-aligned neighbors

                    % --------------------------------------------------
                    % CASE 3: Vertical neighbor  (di=0, dj=±1)
                    % --------------------------------------------------
                    elseif di == 0 && dj ~= 0
                        % X-momentum / u equation:
                        %   Second-order: -C2_loc (from d^2u/dy^2, mu coefficient)
                        %   Gradient correction: +sign(dj) * Gy_mu
                        %     (dmu/dy * du/dy term)
                        append(u_row, u_col, -C2_loc + dj * Gy_mu);

                        % Y-momentum / v equation:
                        %   Second-order: -C1_loc (from d^2v/dy^2)
                        %   Gradient correction: +sign(dj) * Gy_lam2mu
                        %     (d(lam+2mu)/dy * dv/dy term)
                        append(v_row, v_col, -C1_loc + dj * Gy_lam2mu);

                        % No u-v or v-u coupling for axis-aligned neighbors

                    % --------------------------------------------------
                    % CASE 4: Diagonal neighbor  (di=±1, dj=±1)
                    % --------------------------------------------------
                    else
                        % sign product  s = di*dj is ±1, used for both the
                        % base C3 term and the gradient corrections.
                        s = di * dj;

                        % X-momentum equation couples to v at diagonal neighbor:
                        %   Base cross-derivative:  s * C3_loc
                        %   Gradient corrections from:
                        %     dl/dx * dv/dy   -> di * Gx_lam  (sign from dl/dx diff)
                        %     dmu/dy * dv/dx  -> dj * Gy_muC  (sign from dmu/dy diff)
                        %   Combined correction sign:
                        %     The (i+di, j+dj) corner gets +di from dl/dx stencil
                        %     and +dj from dmu/dy stencil (both central diffs).
                        append(u_row, v_col, s*C3_loc + di*Gx_lam/h + dj*Gy_muC/h);

                        % Y-momentum equation couples to u at diagonal neighbor:
                        %   Base cross-derivative:  s * C3_loc
                        %   Gradient corrections from:
                        %     dl/dy * du/dx   -> dj * Gy_lam
                        %     dmu/dx * du/dy  -> di * Gx_muC
                        append(v_row, u_col, s*C3_loc + dj*Gy_lam/h + di*Gx_muC/h);
                    end

                end % di
            end % dj

        end % i
    end % j

    % Trim unused pre-allocated space and build sparse matrix
    I = I(1:ptr);
    J = J(1:ptr);
    V = V(1:ptr);
    K = sparse(I, J, V, total_dofs, total_dofs);

    % ------------------------------------------------------------------
    % 4. Apply Dirichlet BC: zero displacement on all four walls
    %    (modify rows in-place: zero the row, put 1 on diagonal, zero b)
    % ------------------------------------------------------------------
    bc_dofs = [];

    for j = 1:ny
        for i = 1:nx
            if i == 1 || i == nx || j == 1 || j == ny
                nd = getNode(i, j);
                bc_dofs = [bc_dofs, getUdof(nd), getVdof(nd)]; %#ok<AGROW>
            end
        end
    end

    bc_dofs = unique(bc_dofs);
    K(bc_dofs, :)          = 0;
    K(bc_dofs, bc_dofs)    = speye(length(bc_dofs));
    b(bc_dofs)             = 0;

end
