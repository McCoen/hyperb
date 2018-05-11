%source("damping/newton.m");
source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

global boundary = @(x) 0.1 * sin(2 * pi * x);

global task = 2;

global n = 20;
global k = 50;
global vareps = power(10, -16);

global num_of_dampers = 4;
global damper_x = [0.2, 0.4, 0.6, 0.8];
global wt_bounds = [nan, nan; nan, nan; nan, nan; nan, nan];
global st_variation_bounds = [0.0, 0.0];

global enable_interpolation = false;
global full_diff = true;

a = 1;
l = 1;
t = 0.05;

hx = l / n;
ht = t / k;

global eta = power(10, -3);
global xi = power(10, -4);

print_task_init(a, l, t, hx, ht);

u = osc_y(hx);

wt = [];
continue_minimization = false;
if (continue_minimization)
	wt = read_wt_from_file("mqlast.txt");
else
	wt = zeros(k + 1, num_of_dampers);
endif

wt = zeros(k + 1, num_of_dampers);
mq_max_iterations = 1000;
output_filename = "mqlast";
wt = gpuMarquardtMinimization(damper_x, wt_bounds, wt, st_variation_bounds, num_of_dampers, n, k, u, a, l, t, mq_max_iterations, vareps, output_filename);
save_wt_to_file(wt, 'marquardt_last.txt');
%5.250639e-013