source('damping/hyperb.m');

global print_in_color = true;

function print_err(a, l, t, wt_filename, nth)
	global n k num_of_dampers damper_x wt_bounds st_variation_bounds max_abs_wt max_err;

	hx = l / n;
	ht = t / k;

	u = osc_u(hx);
	v = osc_v(a, l, hx);

	wt = read_wt_from_file(wt_filename);
	for i = 1 : 4
		max_abs = 0;
		for j = 1 : k + 1
			if (abs(wt(j, i)) > max_abs)
				max_abs = abs(wt(j, i));
			endif
		endfor
		max_abs_wt(nth, i) = max_abs;
	endfor

	err = gpuEnergyInt(damper_x, wt, st_variation_bounds, num_of_dampers, n, k, u, v, a, l, t);
	max_err(nth) = err;

	disp('E(T) = ');
	disp(err);
	disp('');
endfunction

global h0 = @(x) 0.1 * sin(2 * pi * x);
global h1 = @(x) 0.0;

global task = 3;

global n = 20;
global k = 50;
global vareps = power(10, -16);

global num_of_dampers = 4;
global damper_x = [0.2, 0.4, 0.6, 0.8];
global wt_bounds = [nan, nan; nan, nan; nan, nan; nan, nan];
global st_variation_bounds = [0.0, 0.0];

a = 1;
l = 1;

global max_abs_wt = zeros(37, num_of_dampers);
global max_err = zeros(37, 1);

nth = 1;
for t = 0.01 : 0.0025 : 0.1
	print_err(a, l, t, sprintf('task2_4dampers_%f_wt.txt', t), nth);
	nth = nth + 1;
endfor

disp(max_abs_wt);
disp(max_err);

figure;
xf = [0.01 : power(10, -6) : 0.1];
xp = [0.01 : 0.0025 : 0.1];
spl_a = interp1(xp, max_abs_wt(:, 1), xf, "spline");
spl_a_d = interp1(xp, max_abs_wt(:, 1), xp, "spline");
spl_b = interp1(xp, max_abs_wt(:, 2), xf, "spline");
spl_b_d = interp1(xp, max_abs_wt(:, 2), xp, "spline");
spl_c = interp1(xp, max_abs_wt(:, 3), xf, "spline");
spl_c_d = interp1(xp, max_abs_wt(:, 3), xp, "spline");
spl_d = interp1(xp, max_abs_wt(:, 4), xf, "spline");
spl_d_d = interp1(xp, max_abs_wt(:, 4), xp, "spline");
cmap = hsv(4);
plot(xf, spl_a, 'k', xf, spl_b, '--3;;k', xp, spl_c_d, '-*3;;k', xp, spl_d_d, '-d;;k');
grid on;
legend('W(t)_1 ', 'W(t)_2 ', 'W(t)_3 ', 'W(t)_4 ');
xlabel 't';
ylabel 'Max W(t)';
title 'T from 0.01 to 0.1';
print(sprintf('tasks/task%d_wt.eps', task), '-S480, 320');

figure;
xf = [0.01 : power(10, -6) : 0.1];
xp = [0.01 : 0.0025 : 0.1];
spl_a = interp1(xp, max_err(:, 1), xf, "spline");
spl_a_d = interp1(xp, max_err(:, 1), xp, "spline");
cmap = hsv(4);
plot(xf, spl_a, 'k', xp, spl_a_d, '*3;;b');
grid on;
%legend('Error ');
xlabel 't';
ylabel 'Error';
title 'T from 0.01 to 0.1';
print(sprintf('tasks/task%d_error.eps', task), '-S480, 320');