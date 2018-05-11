source('damping/hyperb.m');

global h0 = @(x) 0.2 * x * (1 - x);
global h1 = @(x) 0.0;

global task = 4;

global n = 20;
global k = 120;
global vareps = power(10, -16);

global num_of_dampers = 2;
global damper_x = [0.25, 0.75];
global wt_bounds = [-2, 2; -2, 2];

global st = zeros(k + 1, num_of_dampers);

a = 1;
l = 1;
t = 0.1825;

hx = l / n;
ht = t / k;

print_task_init(a, l, t, hx, ht);

u = osc_u(hx);
v = osc_v(a, l, hx);

launch_minimization = false;
show_or_print_grid_and_wt = true;

if (launch_minimization)
	wt = zeros(k + 1, num_of_dampers);
	mq_max_iterations = inf;
	wt = gpuMarquardtMinimization(damper_x, wt_bounds, wt, st, num_of_dampers, n, k, u, v, a, l, t, mq_max_iterations, vareps);
	save_wt_to_file(wt, 'output/task3_2dampers_pen_wt.txt');
endif

if (show_or_print_grid_and_wt)

	print_to_eps = true;

	wt = read_wt_from_file('output/task3_2dampers_pen_wt.txt');
	err = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);
	u = gpuOscDeadening(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);


	disp('E(T) = ');
	disp(err);
	disp('');

	%plot_grid_black_and_white(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
	plot_wt_black_and_white(wt, [nan, nan], t, '-S480, 320', print_to_eps);
endif