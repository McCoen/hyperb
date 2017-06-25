source("damping/newton.m");
source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

global boundary = @(x) 0.1 * sin(2 * pi * x);

global task = 2;

global n = 80;
global k = 32;
global vareps = power(10, -6);

global num_of_dampers = 2;
global damper_x = [0.25, 0.75];
global wt_bounds = [nan, nan; nan, nan];
global wt_start = zeros(k + 1, num_of_dampers);

global st = zeros(k + 1, num_of_dampers);
st_max_variation = 0.0;
st_min_variation = 0.0;
for i = 1 : num_of_dampers
	for j = 1 : k + 1
		if (mod(j, 2) == 1)
			st(j, i) = st_max_variation;
		else
			st(j, i) = st_min_variation;
		endif
	endfor
endfor
disp(st);

global enable_interpolation = false;
global full_diff = true;

global a = 1;
global l = 1;
t = 0.4;

hx = l / n;
ht = t / k;

global eta = power(10, -3);
global xi = power(10, -4);

print_task_init(t, hx, ht);

u = osc_y(hx);

wt = gpuMarquardt(damper_x, wt_start, st, num_of_dampers, n, k, u, 1, 1, t, 1, vareps);
disp(wt);
disp('');

%u = oscDeadening(damper_x, wt, st, n, k, u, 1, 1, t, 0, vareps);
%extended_print_grid_to_file(u, t, n, k, hx, ht, "test.m");
%disp(u);
%disp('');
exit;
