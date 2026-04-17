function tmp_trace_amg_issue
nx=30; ny=30; h=1/(nx+1);
x_vec=linspace(h,1-h,nx); y_vec=linspace(h,1-h,ny);
[X,Y]=meshgrid(x_vec,y_vec);
mu_bg=26; lambda_bg=51; mu_inc=77; lambda_inc=115;
r_inc=0.2; cx=0.5; cy=0.5;
in_inclusion=((X-cx).^2+(Y-cy).^2)<=r_inc^2;
mu_arr=mu_bg*ones(ny,nx); lambda_arr=lambda_bg*ones(ny,nx);
mu_arr(in_inclusion)=mu_inc; lambda_arr(in_inclusion)=lambda_inc;
fx=zeros(ny,nx); fy=-ones(ny,nx);
[K,b]=build_navier_cauchy_heterogeneous(nx,ny,h,lambda_arr,mu_arr,fx,fy);
hier=amg_setup(K,0.25,8,40);
trace_level(hier,1,b,zeros(size(b)),1,1);
end

function x=trace_level(hierarchy,lev,b,x,nu1,nu2)
A=hierarchy{lev}.A;
fprintf('enter lev %d n %d rhsnorm %.3e\n',lev,size(A,1),norm(b));
if lev==length(hierarchy) || isempty(hierarchy{lev}.P)
    x=A\b;
    fprintf('coarse lev %d sol nan %d inf %d norm %.3e rcond %.3e\n', ...
        lev,any(isnan(x)),any(isinf(x)),norm(x),rcond(full(A)));
    return;
end

x=gs(A,b,x,nu1,'forward');
fprintf(' after pre GS lev %d nan %d inf %d norm %.3e\n',lev,any(isnan(x)),any(isinf(x)),norm(x));
r=b-A*x;
fprintf(' residual lev %d nan %d inf %d norm %.3e\n',lev,any(isnan(r)),any(isinf(r)),norm(r));
rc=hierarchy{lev}.R*r;
fprintf(' coarse rhs lev %d nan %d inf %d norm %.3e\n',lev,any(isnan(rc)),any(isinf(rc)),norm(rc));
ec=trace_level(hierarchy,lev+1,rc,zeros(size(rc)),nu1,nu2);
fprintf(' returned ec lev %d nan %d inf %d norm %.3e\n',lev+1,any(isnan(ec)),any(isinf(ec)),norm(ec));
x=x+hierarchy{lev}.P*ec;
fprintf(' after corr lev %d nan %d inf %d norm %.3e\n',lev,any(isnan(x)),any(isinf(x)),norm(x));
x=gs(A,b,x,nu2,'backward');
fprintf(' after post GS lev %d nan %d inf %d norm %.3e\n',lev,any(isnan(x)),any(isinf(x)),norm(x));
end

function x=gs(A,b,x,nu,direction)
n=size(A,1);
d=diag(A);
if strcmpi(direction,'forward')
    ord=1:n;
else
    ord=n:-1:1;
end

for sweep=1:nu
    for ii=ord
        residual_i=b(ii)-A(ii,:)*x+d(ii)*x(ii);
        x(ii)=residual_i/d(ii);
        if ~isfinite(x(ii))
            fprintf(' nonfinite at row %d diag %.3e residual %.3e\n',ii,d(ii),full(residual_i));
            return;
        end
    end
end
end
