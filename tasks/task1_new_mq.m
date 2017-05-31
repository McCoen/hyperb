source("minimization/newton.m");
source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

global boundary = @(x) 0.25 * sin(pi * x);

global task = 1;

global a = 1;
global l = 1;

global vareps = power(10, -6);

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan];

global enable_interpolation = false;
global full_diff = true;

global n = 160;
global k = 64;
global wt_start = zeros(k + 1, num_of_dampers);

t = 0.08;
hx = l / n;
ht = t / k;

global eta = power(10, -3);
global xi = power(10, -4);

print_task_init(t, hx, ht);

%wt = marquardt_continue(@solve_all, damper_x, 'marquardt_last', 5, vareps);
alpha = osc_alpha(hx, ht);
beta = osc_beta(hx);

b = osc_b();
c = osc_c(alpha, b);
c_inverted = inv(c);
sweep_side = eye(2);

cTilde = osc_cTilde(alpha, b);

u = osc_y(hx);
v = osc_sec_y();

global wt = wt_start;
for i = 1 : length(wt(:))
	wt(i) = 1;%(rand(1) - rand(1)) * 10;
endfor
print_dampers_info();

# wt = marquardt(@solve_all, damper_x, wt, 1500, vareps, power(10, 4), hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta, false);

print_dampers_info();
disp('E(T) = ');
alpha = osc_alpha(hx, ht);
beta = osc_beta(hx);

u = oscDeadening(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
err = energyInt(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
disp(err);
disp('');

print_u_grid_values(ht, u);

%plot_u_grid(err, t, hx, ht, u);
#plot_f(1, damper_x, wt);
%plot_wt(wt, 1);
