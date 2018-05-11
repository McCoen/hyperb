source('damping/hyperb.m');

global h0 = @(x) 0.1 * sin(3 * pi * x);
global h1 = @(x) 0.0;

global task = 6;

global n = 20;
global k = 20;
global vareps = power(10, -16);

global num_of_dampers = 1;
global damper_x = [nan];
global wt_bounds = [nan, nan];

global st = zeros(k + 1, num_of_dampers);

global is_empirical;

a = 1;
l = 1;
t = 0.5;

hx = l / n;
ht = t / k;

print_task_init(a, l, t, hx, ht);

u = osc_u(hx);
v = osc_v(a, l, hx);
%disp(v);
%v = zeros(k + 1, n + 1);

empirical_approach = false;
launch_minimization = false;
show_or_print_grid_and_wt = true;

wt = zeros(k + 1, num_of_dampers);
if (launch_minimization)
	mq_max_iterations = inf;
	wt = gpuMarquardtMinimization(damper_x, wt_bounds, wt, st, num_of_dampers, n, k, u, v, a, l, t, mq_max_iterations, vareps);
	save_wt_to_file(wt, 'output/task6_0_5_const_wt.txt');
endif

if (show_or_print_grid_and_wt)
	is_empirical = false;

	print_to_eps = true;

	%wt = read_wt_from_file('output/task1_2dampers_0_5_wt_0.txt');
	wt = read_wt_from_file('output/task6_0_5_const_wt.txt');
	err = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);
	u = gpuOscDeadening(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);

	disp('E(T) = ');
	disp(err);
	disp('');

	%plot_grid_black_and_white(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
	%plot_wt_black_and_white(wt, wt_bounds, t, '-S480, 320', print_to_eps);
	plot_grid(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
	plot_wt_black_and_white(wt, wt_bounds, t, '-S480, 320', print_to_eps);

	%disp(u);
endif