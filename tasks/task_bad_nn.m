source("damping/newton.m");
source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

global boundary = @(x) 0.1 * sin(2 * pi * x);

global task = 1;

global a = 1;
global l = 1;


%  1.4138e-06 0.2
%  1.4269e-06 0.19
%  7.5360e-06 0.15
%  4.9173e-04 0.12
%  0.0072092 0.1

global t_a = 0.1;
global t_b = 0.2;
%  global t_total = 0.1;

global vareps = power(10, -6);

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan; nan, nan];

global enable_interpolation = false;
global full_diff = true;

global n = 16;
global k = 8;
global wt_start = zeros(k + 1, num_of_dampers);
%  t = newton(0.1, 0.2, damper_x, wt_start);
t = 0.1;
hx = l / n;
ht = t / k;

global eta = power(10, -3);
global xi = power(10, -4);

print_task_init(t, hx, ht);

%plot_start_h(1);
%exit;

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

global wt = [0
0
0
0
0
0
0
0
0];
%  wt = marquardt(@solve_all, damper_x, wt_start, 100, vareps, power(10, 4), hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta, false);

%  disp(ei);
%  ei = power(10, -4) - solve_all(wt, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
%  wt = marquardt(@solve_all, damper_x, wt_start, 100, vareps, power(10, 4), false);

print_dampers_info();
disp('E(T) = ');
alpha = osc_alpha(hx, ht);
beta = osc_beta(hx);

%  	wt = marquardt(@solve_all, damper_x, wt_start, 100, vareps, power(10, 4), hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta, true);
	%err = power(10, -4) - solve_all(wt, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	
u = oscDeadening(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
err = energyInt(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
extended_print_grid_to_file(u, n, k, hx, ht, "absOut");
disp(err);
%printf("%.16f\n", err);
disp('');

print_u_grid_values(ht, u);

%plot_u_grid(err, t, hx, ht, y);
#plot_f(1, damper_x, wt);
%plot_wt(wt, 1);
