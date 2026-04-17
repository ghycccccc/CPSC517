function z = geomg_preconditioner(hierarchy, r, nu1, nu2)
% GEOMG_PRECONDITIONER
% Apply one geometric multigrid V-cycle from zero initial guess.

    z = geomg_vcycle(hierarchy, 1, r, zeros(size(r)), nu1, nu2);
end


function x = geomg_vcycle(hierarchy, lev, b, x, nu1, nu2)
    A = hierarchy{lev}.A;

    if lev == length(hierarchy) || isempty(hierarchy{lev}.P)
        x = A \ b;
        return;
    end

    P = hierarchy{lev}.P;
    R = hierarchy{lev}.R;

    x = gs_smoother(A, b, x, nu1, 'forward');
    r = b - A * x;
    r_c = R * r;

    e_c = geomg_vcycle(hierarchy, lev + 1, r_c, zeros(size(r_c)), nu1, nu2);
    x = x + P * e_c;
    x = gs_smoother(A, b, x, nu2, 'backward');
end


function x = gs_smoother(A, b, x, nu, direction)
    d = diag(A);
    L = tril(A, -1);
    U = triu(A, 1);

    if strcmpi(direction, 'forward')
        sweep_order = 1:size(A, 1);
    else
        sweep_order = size(A, 1):-1:1;
    end

    for sweep = 1:nu
        for i = sweep_order
            x(i) = (b(i) - L(i, :) * x - U(i, :) * x) / d(i);
        end
    end
end
