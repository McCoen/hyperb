source("damping/newton.m");
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

global t_a = 0.1;
global t_b = 0.2;

global vareps = power(10, -6);

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan; nan, nan];

global enable_interpolation = false;
global full_diff = true;

global n = 160;
global k = 64;
global wt_start = zeros(k + 1, num_of_dampers);

t = 1.0;

hx = l / n;
ht = t / k;

global eta = power(10, -3);
global xi = power(10, -4);

print_task_init(t, hx, ht);

alpha = osc_alpha(hx, ht);
beta = osc_beta(hx);

b = osc_b();
c = osc_c(alpha, b);
c_inverted = inv(c);
sweep_side = eye(2);

cTilde = osc_cTilde(alpha, b);

u = osc_y(hx);
v = osc_sec_y();

alpha = osc_alpha(hx, ht);
beta = osc_beta(hx);

wt = wt_start;
wt(1) = 7.5;
wt(2) = -97.5;
disp(wt);
grad = zeros(1, 1);
wt = gpuMarquardt(damper_x, wt, n, k, u, 1, 1, 1);
%disp(wt);
exit;
u = oscDeadening(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
print_u_grid_values(ht, u);
do
	err = energyInt(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
until (err != -1)
disp('E(T) = ');
disp(err);
disp('');

%extended_print_grid_to_file(u, n, k, hx, ht, "absOut");

%plot_u_grid(err, t, hx, ht, y);
#plot_f(1, damper_x, wt);
%plot_wt(wt, 1);
