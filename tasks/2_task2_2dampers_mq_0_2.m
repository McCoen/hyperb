source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

function boundary = boundary(x)
	boundary = 0.1 * sin(2 * pi * x);
endfunction

global task = 2;

global a = 1;
global l = 1;
global t_total = 0.2;

global vareps = power(10, -10);

global n = 32;
global k = 16;

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan];

global enable_interpolation = true;
global full_diff = true;

global hx = l / n;
global ht = t_total / k;
global wt = zeros(k + 1, num_of_dampers);

print_task_init();

wt = marquardt(@solve_all, damper_x, wt, Inf, vareps, power(10, 4));

print_dampers_info();
disp('E(T) = ');
err = solve_all(damper_x, wt);
disp(err);
disp('');

print_u_grid_values();

plot_u_grid(err);
#plot_f(1, damper_x, wt);
%plot_wt(wt, 1);
