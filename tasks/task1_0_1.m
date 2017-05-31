source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");

function boundary = boundary(x)
	boundary = 0.25 * sin(pi * x);
endfunction

global task = 1;

global a = 1;
global l = 1;
global t_total = 0.1;

global vareps = power(10, -10);

global n = 128;
global k = 64;

global num_of_dampers = 2;
global damper_x = [0.5, 0.75];
global wt_bounds = [nan, nan; nan, nan];

global enable_interpolation = true;
global full_diff = true;

global hx = l / n;
global ht = t_total / k;

global wt = zeros(k + 1, num_of_dampers);

print_task_init();

%plot_start_h(1);
%exit;

print_dampers_info();
disp('E(T) = ');
err = solve_all();
disp(err);
disp('');

print_u_grid_values();

%plot_u_grid(err);
#plot_f(1, damper_x, wt);
%plot_wt(wt, 1);