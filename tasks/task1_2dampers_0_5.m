source('damping/hyperb.m');

global h0 = @(x) 0.1 * sin(2 * pi * x);
global h1 = @(x) 0.0;

global task = 1;

global n = 20;
global k = 250;
global vareps = power(10, -16);

global num_of_dampers = 2;
global damper_x = [0.25, 0.75];
global wt_bounds = [nan, nan; nan, nan];

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

empirical_approach = true;
launch_minimization = false;
show_or_print_grid_and_wt = true;

wt = zeros(k + 1, num_of_dampers);
if (launch_minimization)
	mq_max_iterations = inf;
	wt = gpuMarquardtMinimization(damper_x, wt_bounds, wt, st, num_of_dampers, n, k, u, v, a, l, t, mq_max_iterations, vareps);
	save_wt_to_file(wt, 'output/task1_2dampers_0_5_wt_0.txt');
endif

if (show_or_print_grid_and_wt)
	is_empirical = false;

	print_to_eps = false;

	wt = read_wt_from_file('output/task1_2dampers_0_5_wt_0.txt');
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

u = osc_u(hx);
v = osc_v(a, l, hx);

wt = zeros(k + 1, num_of_dampers);
if (empirical_approach)
	is_empirical = true;

	et = [1000	1000
		-1.1959e-01	-2.7326e+00
		3.1708e-02	3.2003e+00
		1000	1000
		-6.6844e+00	-2.5081e+00
		2.9977e+00	-3.1785e+00
		-6.5086e+00	9.9958e+00];

	wt = empirical_wt(et, t);

	if (show_or_print_grid_and_wt)
		print_to_eps = false;

		err = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);
		u = gpuOscDeadening(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);

		disp('E(T) = ');
		disp(err);
		disp('');

		%plot_grid_black_and_white(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
		plot_wt_black_and_white(wt, wt_bounds, t, '-S480, 320', print_to_eps);
	endif
endif

if (launch_minimization)
	mq_max_iterations = inf;
	wt = gpuMarquardtMinimization(damper_x, wt_bounds, wt, st, num_of_dampers, n, k, u, v, a, l, t, mq_max_iterations, vareps);
	save_wt_to_file(wt, 'output/task1_2dampers_0_5_wt.txt');
endif

if (show_or_print_grid_and_wt)
	is_empirical = false;

	print_to_eps = true;

	wt = read_wt_from_file('output/task1_2dampers_0_5_wt.txt');
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