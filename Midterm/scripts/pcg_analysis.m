nx = 20; ny = 20; h = 0.1;
lambda = 1; mu = 1;
fx = rand(nx, ny); fy = rand(nx, ny);

% Build matrix with "Node" ordering
[K, b] = build_navier_cauchy(nx, ny, h, lambda, mu, fx, fy, 'node');
K(1, :) = zeros(1, nx * ny * 2);
K(2, :) = zeros(1, nx * ny * 2);
K(1, 1) = 1.0;
K(2, 2) = 1.0;
b(1) = 0;
b(2) = 0;

% VERIFY: Is the matrix actually SPD?
% Try a direct solve first. If this fails, the matrix is singular.
try
    x_direct = K \ b;
    disp('Direct solver succeeded. Matrix is non-singular.');
catch
    disp('Direct solver failed. Matrix is singular! Check BCs.');
end

% Now run your comparison
compare_preconditioners(K, b);