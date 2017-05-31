source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

global boundary = @(x) 0.2 * sin(2 * pi * x);

global task = 3;

global a = 1;
global l = 1;
global t_total = 0.05;

global vareps = power(10, -12);

global num_of_dampers = 1;
global damper_x = [0.25];
global wt_bounds = [nan, nan];

global enable_interpolation = false;
global full_diff = true;

global n = 16;
global k = 8;
global hx = l / n;
global ht = t_total / k;
global wt = zeros(k + 1, num_of_dampers);

global alpha = osc_alpha();
global beta = osc_beta();

global b = osc_b();
global c = osc_c();
global c_inverted = inv(c);
global sweep_side = eye(2);
global cTilde = osc_cTilde();

global y = osc_y();
global y_sec = osc_sec_y();

global xr = 0.0 : hx : l;
global yr = 0.0 : ht : t_total;

global eta = power(10, -3);
global xi = power(10, -4);

print_task_init();

%  plot_start_h();
%  exit;

#wt = marquardt_continue(@solve_all, damper_x, 'marquardt_last', 10, vareps);
%wt = marquardt(@solve_all, damper_x, wt, 10, vareps, power(10, 4));

wt = [-5.6869e+03;
   4.0961e+03;
  -3.1850e+03;
   1.2350e+03;
   1.9902e+03;
  -4.1672e+03;
   8.2369e+03;
  -1.2230e+04;
   1.5602e+04];

print_dampers_info();
disp('E(T) = ');
err = solve_all(wt);
disp(err);
disp('');

print_u_grid_values();

%  plot_u_grid(err);
#plot_f(1, damper_x, wt);
plot_wt(wt, 1);
