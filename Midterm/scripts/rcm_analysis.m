N = 20;
h = 0.1;
mu = 1.0;
lambda = 1.0;
fx = ones(N);


fy = ones(N);

figure;

[K_n, b_n] = build_navier_cauchy(N, N, h, lambda, mu, fx, fy, "node");
subplot(2, 2, 1);
spy(K_n);                      % Figure 1: Sparsity pattern

p_rcm_n = symrcm(K_n);
subplot(2, 2, 2);
spy(K_n(p_rcm_n, p_rcm_n));         % Figure 2: RCM reordering
%L = chol(K(p_rcm, p_rcm));
%spy(L);                      % Figure 3: Fill-in after factorization

[K_c, b_c] = build_navier_cauchy(N, N, h, lambda, mu, fx, fy, "component");
subplot(2, 2, 3);
spy(K_c);                      % Figure 1: Sparsity pattern

p_rcm_c = symrcm(K_c);
subplot(2, 2, 4);
spy(K_c(p_rcm_c, p_rcm_c));         % Figure 2: RCM reordering
%L = chol(K(p_rcm, p_rcm));
%spy(L);                      % Figure 3: Fill-in after factorization