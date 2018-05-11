source('damping/hyperb.m');

global h0 = @(x) 0.25 * exp (x) * sin (2 * pi * x);
global h1 = @(x) 0.0;

global task = 5;

global n = 20;
global k = 250;
global vareps = power(10, -16);

global num_of_dampers = 2;
global damper_x = [0.25, 0.75];
global wt_bounds = [nan, nan; nan, nan];

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
	save_wt_to_file(wt, 'output/task4_2dampers_0_1_1_wt.txt');
endif

if (show_or_print_grid_and_wt)

	print_to_eps = true;

	wt = read_wt_from_file('output/task4_2dampers_0_1_1_wt.txt');
	err = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);
	u = gpuOscDeadening(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);


	disp('E(T) = ');
	disp(err);
	disp('');

	%plot_grid_black_and_white(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
	%plot_wt_black_and_white(wt, wt_bounds, t, '-S480, 320', print_to_eps);
	plot_grid(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
endif

figure;
xf = [0 : power(10, -4) : 1.0];
xp = [0 : 1 / n : 1.0];
wt_nth = ones(n + 1, 1);
for x = 1 : n + 1
	wt_nth(x) = u(k + 1, x);
end
spl = interp1(xp, wt_nth, xf, 'spline');
	
dp = [-1.0 * power(10, -2) : 2.0 * power(10, -2) / 40 : 1.0 * power(10, -2)];
d_nth = ones(41, 1);
	
for x = 1 : 41
	d_nth(x) = damper_x(1);
end
spl_d = interp1(dp, d_nth, dp, 'spline');

spl_u_d = interp1(xp, wt_nth, xp, 'spline');

d2_nth = ones(41, 1);

for x = 1 : 41
	d2_nth(x) = damper_x(2);
end
spl_d2 = interp1(dp, d2_nth, dp, 'spline');
	
plot(xf, spl, 'm', xp, spl_u_d, '*3;;b', spl_d, dp, '+3;;k', spl_d2, dp, '+3;;k');
grid on;
xlabel 'X-axis';
ylabel 'U-axis';
title (sprintf('U(x, T = %f)', t));
print(sprintf('tasks/task%d_%ddampers_t_%f_cut_1.eps', task, num_of_dampers, t), '-color', '-S480, 320');