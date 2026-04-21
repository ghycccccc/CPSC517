function [K, b] = build_navier_cauchy_heterogeneous(nx, ny, h, lambda, mu, fx, fy)
% BUILD_NAVIER_CAUCHY_HETEROGENEOUS  (corrected)
%   Assembles K and b for 2D heterogeneous Navier-Cauchy via central FD.
%
%   KEY CORRECTION over previous version:
%   The terms  dl/dx * dv/dy  and  dmu/dy * dv/dx  in the x-momentum
%   equation (and symmetric terms in y-momentum) produce u-v coupling
%   at AXIS-ALIGNED neighbors. This was previously missing entirely.
%   The corrected stencil per interior node has up to 18 nonzero columns
%   per equation row (vs 9 in the homogeneous case).
%
%   Also fixed: extra spurious /h factor on diagonal gradient corrections.
%
% INPUTS:
%   nx, ny  - grid size
%   h       - uniform grid spacing
%   lambda  - [ny x nx]  first Lame parameter
%   mu      - [ny x nx]  shear modulus
%   fx, fy  - [ny x nx]  body force components
%
% OUTPUT:
%   K  - sparse [2*nx*ny x 2*nx*ny]
%   b  - [2*nx*ny x 1]

    assert(all(size(lambda) == [ny, nx]), 'lambda must be [ny x nx]');
    assert(all(size(mu)     == [ny, nx]), 'mu must be [ny x nx]');
    assert(all(size(fx)     == [ny, nx]), 'fx must be [ny x nx]');
    assert(all(size(fy)     == [ny, nx]), 'fy must be [ny x nx]');

    N          = nx * ny;
    total_dofs = 2 * N;

    getNode = @(i, j) (j - 1) * nx + i;
    getUdof = @(nd)    2 * nd - 1;
    getVdof = @(nd)    2 * nd;

    % ------------------------------------------------------------------
    % Material gradients  [ny x nx], central diff interior, one-sided BC
    % ------------------------------------------------------------------
    dlambda_dx = zeros(ny, nx);   dlambda_dy = zeros(ny, nx);
    dmu_dx     = zeros(ny, nx);   dmu_dy     = zeros(ny, nx);

    dlambda_dx(:, 2:nx-1) = (lambda(:,3:nx  )-lambda(:,1:nx-2))/(2*h);
    dlambda_dy(2:ny-1, :) = (lambda(3:ny,:  )-lambda(1:ny-2,:))/(2*h);
    dmu_dx    (:, 2:nx-1) = (mu    (:,3:nx  )-mu    (:,1:nx-2))/(2*h);
    dmu_dy    (2:ny-1, :) = (mu    (3:ny,:  )-mu    (1:ny-2,:))/(2*h);

    dlambda_dx(:, 1 ) = (lambda(:,2   )-lambda(:,1   ))/h;
    dlambda_dx(:, nx) = (lambda(:,nx  )-lambda(:,nx-1))/h;
    dlambda_dy(1,  :) = (lambda(2,:   )-lambda(1,:   ))/h;
    dlambda_dy(ny, :) = (lambda(ny,:  )-lambda(ny-1,:))/h;
    dmu_dx(:, 1 ) = (mu(:,2   )-mu(:,1   ))/h;
    dmu_dx(:, nx) = (mu(:,nx  )-mu(:,nx-1))/h;
    dmu_dy(1,  :) = (mu(2,:   )-mu(1,:   ))/h;
    dmu_dy(ny, :) = (mu(ny,:  )-mu(ny-1,:))/h;

    % ------------------------------------------------------------------
    % Triplet assembly.  Each interior node: up to 18 entries per row.
    % ------------------------------------------------------------------
    est_nnz = 36 * N;     % 18 per equation x 2 equations, conservative
    TI = zeros(est_nnz,1);  TJ = zeros(est_nnz,1);  TV = zeros(est_nnz,1);
    ptr = 0;
    b   = zeros(total_dofs, 1);

    function append(row, col, val)
        if val == 0, return; end   % skip exact zeros (saves fill)
        ptr     = ptr + 1;
        TI(ptr) = row;
        TJ(ptr) = col;
        TV(ptr) = val;
    end

    for j = 1:ny
        for i = 1:nx

            lam = lambda(j,i);   muu = mu(j,i);

            % ---- Second-order coefficients (node-local) ----------------
            C1 = (lam + 2*muu) / h^2;
            C2 =  muu          / h^2;
            C3 = (lam +  muu ) / (4*h^2);

            % ---- Material gradients at this node -----------------------
            dLdx = dlambda_dx(j,i);   dLdy = dlambda_dy(j,i);
            dMdx = dmu_dx    (j,i);   dMdy = dmu_dy    (j,i);

            % ---- Gradient correction scalars ---------------------------
            % All have units [Pa/m / m] = [Pa/m^2] = same as C1,C2,C3.
            % Factor 1/4 comes from product of two central-diff 1/(2h) terms.
            %
            % SAME-DOF axis-aligned corrections:
            %   u-eqn / u-nbr horizontal:  d(lam+2mu)/dx * du/dx  -> ±(dLdx+2dMdx)/4
            %   v-eqn / v-nbr horizontal:  d(mu)/dx     * dv/dx  -> ±dMdx/4
            %   u-eqn / u-nbr vertical:    d(mu)/dy     * du/dy  -> ±dMdy/4
            %   v-eqn / v-nbr vertical:    d(lam+2mu)/dy* dv/dy  -> ±(dLdy+2dMdy)/4
            Ax_L2M = (dLdx + 2*dMdx) / 4;
            Ax_M   =  dMdx           / 4;
            Ay_M   =  dMdy           / 4;
            Ay_L2M = (dLdy + 2*dMdy) / 4;

            % CROSS-DOF axis-aligned corrections (PREVIOUSLY MISSING):
            %   u-eqn / v-nbr horizontal:  d(mu)/dy  * dv/dx  -> ±dMdy/4
            %   u-eqn / v-nbr vertical:    d(lam)/dx * dv/dy  -> ±dLdx/4
            %   v-eqn / u-nbr horizontal:  d(lam)/dy * du/dx  -> ±dLdy/4
            %   v-eqn / u-nbr vertical:    d(mu)/dx  * du/dy  -> ±dMdx/4
            Ax_M_uv  = dMdy / 4;   % u-eqn, v at horizontal neighbor
            Ay_L_uv  = dLdx / 4;   % u-eqn, v at vertical neighbor
            Ax_L_vu  = dLdy / 4;   % v-eqn, u at horizontal neighbor
            Ay_M_vu  = dMdx / 4;   % v-eqn, u at vertical neighbor

            % DIAGONAL corrections (no extra 1/h: cross-deriv stencil
            % already contributes 1/(4h^2) which cancels h^2 in C3):
            %   u-eqn / v-corner:  dl/dx*dv/dy -> di*Dx_L, dmu/dy*dv/dx -> dj*Dy_M
            %   v-eqn / u-corner:  dl/dy*du/dx -> dj*Dy_L, dmu/dx*du/dy -> di*Dx_M
            Dx_L = dLdx / 4;   Dy_M = dMdy / 4;
            Dy_L = dLdy / 4;   Dx_M = dMdx / 4;

            % ---- DOF rows -----------------------------------------------
            nd    = getNode(i,j);
            u_row = getUdof(nd);
            v_row = getVdof(nd);

            b(u_row) = fx(j,i);
            b(v_row) = fy(j,i);

            % ==============================================================
            for dj_s = -1:1
                for di_s = -1:1
                    ni = i + di_s;   nj = j + dj_s;
                    if ni<1||ni>nx||nj<1||nj>ny, continue; end

                    nnd   = getNode(ni, nj);
                    u_col = getUdof(nnd);
                    v_col = getVdof(nnd);

                    % --------------------------------------------------
                    if di_s==0 && dj_s==0          % SELF
                    % --------------------------------------------------
                        append(u_row, u_col,  2*(C1+C2));
                        append(v_row, v_col,  2*(C1+C2));
                        % No self u-v coupling (central diff zero at center)

                    % --------------------------------------------------
                    elseif dj_s==0                 % HORIZONTAL  (di=±1)
                    % --------------------------------------------------
                        % Same-DOF second-order + same-DOF gradient correction
                        append(u_row, u_col, -C1 + di_s * Ax_L2M);
                        append(v_row, v_col, -C2 + di_s * Ax_M  );

                        % Cross-DOF gradient correction (previously missing)
                        % u-eqn gets v contribution: dmu/dy * dv/dx
                        append(u_row, v_col,  di_s * Ax_M_uv);
                        % v-eqn gets u contribution: dlambda/dy * du/dx
                        append(v_row, u_col,  di_s * Ax_L_vu);

                    % --------------------------------------------------
                    elseif di_s==0                 % VERTICAL  (dj=±1)
                    % --------------------------------------------------
                        % Same-DOF
                        append(u_row, u_col, -C2 + dj_s * Ay_M  );
                        append(v_row, v_col, -C1 + dj_s * Ay_L2M);

                        % Cross-DOF (previously missing)
                        % u-eqn gets v contribution: dlambda/dx * dv/dy
                        append(u_row, v_col,  dj_s * Ay_L_uv);
                        % v-eqn gets u contribution: dmu/dx * du/dy
                        append(v_row, u_col,  dj_s * Ay_M_vu);

                    % --------------------------------------------------
                    else                           % DIAGONAL  (di=±1, dj=±1)
                    % --------------------------------------------------
                        s = di_s * dj_s;

                        % u-eqn / v-col: base C3 + gradient corrections
                        % FIX: removed spurious /h factor present in old code
                        append(u_row, v_col,  s*C3 + di_s*Dx_L + dj_s*Dy_M);

                        % v-eqn / u-col
                        append(v_row, u_col,  s*C3 + dj_s*Dy_L + di_s*Dx_M);
                    end

                end
            end
            % ==============================================================

        end
    end

    TI = TI(1:ptr);   TJ = TJ(1:ptr);   TV = TV(1:ptr);
    K  = sparse(TI, TJ, TV, total_dofs, total_dofs);

    % ------------------------------------------------------------------
    % Dirichlet BC: zero on all four walls
    % ------------------------------------------------------------------
    bc_dofs = [];
    for j = 1:ny
        for i = 1:nx
            if i==1||i==nx||j==1||j==ny
                nd      = getNode(i,j);
                bc_dofs = [bc_dofs, getUdof(nd), getVdof(nd)]; 
            end
        end
    end
    bc_dofs             = unique(bc_dofs);
    K(bc_dofs,:)        = 0;
    K(bc_dofs,bc_dofs)  = speye(length(bc_dofs));
    b(bc_dofs)          = 0;

end
