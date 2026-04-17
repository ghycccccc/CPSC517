function tmp_compare_cases
run_case(26,51,26,51);
run_case(26,51,77,115);
end

function run_case(mu_bg,lambda_bg,mu_inc,lambda_inc)
nx=30; ny=30; h=1/(nx+1);
x_vec=linspace(h,1-h,nx); y_vec=linspace(h,1-h,ny);
[X,Y]=meshgrid(x_vec,y_vec);
r_inc=0.2; cx=0.5; cy=0.5;
in_inclusion=((X-cx).^2+(Y-cy).^2)<=r_inc^2;
mu_arr=mu_bg*ones(ny,nx); lambda_arr=lambda_bg*ones(ny,nx);
mu_arr(in_inclusion)=mu_inc; lambda_arr(in_inclusion)=lambda_inc;
fx=zeros(ny,nx); fy=-ones(ny,nx);
[K,b]=build_navier_cauchy_heterogeneous(nx,ny,h,lambda_arr,mu_arr,fx,fy);
fprintf('case mu_inc %.1f lambda_inc %.1f symerr %.3e\n', ...
    mu_inc,lambda_inc,norm(K-K','fro')/norm(K,'fro'));
hier=amg_setup(K,0.25,8,40);
z=amg_preconditioner(hier,b,1,1);
fprintf('  prec nan %d norm %.3e levels %d\n',any(isnan(z)),norm(z),length(hier));
[~,flag,relres,it]=gmres(K,b,30,1e-8,10,@(r)amg_preconditioner(hier,r,1,1));
fprintf('  gmres flag %d rel %.3e iter [%d,%d]\n',flag,relres,it(1),it(2));
end
