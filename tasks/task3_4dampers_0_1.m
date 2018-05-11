source('damping/hyperb.m');

global h0 = @(x) 0.1 * sin(2 * pi * x);
global h1 = @(x) 0.0;

global task = 3;

global n = 20;
global k = 50;
global vareps = power(10, -16);

global num_of_dampers = 4;
global damper_x = [0.2, 0.4, 0.6, 0.8];
global wt_bounds = [nan, nan; nan, nan; nan, nan; nan, nan];

global st = zeros(k + 1, num_of_dampers);

a = 1;
l = 1;
t = 0.1;

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
	save_wt_to_file(wt, 'tasks/task3_4dampers_0_1_wt.txt');
endif

if (show_or_print_grid_and_wt)

	print_to_eps = true;

	wt = read_wt_from_file('tasks/task3_4dampers_0_1_wt.txt');
	err = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);
	u = gpuOscDeadening(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);


	disp('E(T) = ');
	disp(err);
	disp('');

	%plot_grid_black_and_white(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
	%plot_wt_black_and_white(wt, wt_bounds, t, '-S480, 320', print_to_eps);
	plot_grid(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
	plot_wt(wt, wt_bounds, t, '-S480, 320', print_to_eps);
endif