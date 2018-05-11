1;

global spl_x;

function save_print_me_script(err, wt, u, t, hx, ht)
	global task num_of_dampers n k;

	output_file = fopen(sprintf('task%d_%ddampers_%f_print_me.m', task, num_of_dampers, t), 'w');
	fprintf(output_file, 'source(''damping/hyperb.m'');\n\n');
	fprintf(output_file, sprintf('a = 1;\nl = 1;\nt = %e;\nhx = %e;\nht = %e;\nerr = %e;\n\n', t, hx, ht, err));

	fprintf(output_file, 'wt = [');
	for i = 1 : k
		for j = 1 : num_of_dampers
			fprintf(output_file, '%e ', wt(i, j));
		endfor
		fprintf(output_file, '\n');
	endfor
	for j = 1 : num_of_dampers
		fprintf(output_file, '%e ', wt(k + 1, j));
	endfor
	fprintf(output_file, '];\n\n');

	fprintf(output_file, 'u = [');
	for i = 1 : k
		for j = 1 : n + 1
			fprintf(output_file, '%e ', u(i, j));
		endfor
		fprintf(output_file, '\n');
	endfor
	for j = 1 : n + 1
			fprintf(output_file, '%e ', u(k + 1, j));
	endfor
	fprintf(output_file, '];\n\n');

	fprintf(output_file, sprintf('plot_u_grid(%e, a, l, t, hx, ht, u, 100, "january 1440 900", "-S1440, 900");\n', err));
	fprintf(output_file, 'pause;\n');

	fclose(output_file);
endfunction

function wt = empirical_wt(et, t)
	global k num_of_dampers;

	wt = zeros(k + 1, num_of_dampers);
	for i = 1 : num_of_dampers
		for j = 1 : k + 1
			currentT = (t / k) * (j - 1);
			wt(j, i) = et(1, i) * sin(et(2, i) * currentT + et(3, i)) + et(4, i) * sin(et(5, i) * currentT + et(6, i)) * sin(et(7, i) * currentT);
		endfor
	endfor
endfunction

function wt = read_wt_from_file(filename)
	global k num_of_dampers;

	str = '';
	wt = zeros(k + 1, num_of_dampers);
	input_file = fopen(filename, 'r');

	for i = 1 : num_of_dampers
		for j = 1 : k + 1
			current = fscanf(input_file, "%s", str);
			wt(j, i) = str2num(current);
			%disp(wt(j, i));
		endfor
	endfor
	fclose(input_file);
endfunction

function wt = save_wt_to_file(wt, filename)
	global k num_of_dampers

	output_file = fopen(filename, 'w');
	for i = 1 : num_of_dampers
		for j = 1 : k + 1
			fprintf(output_file, '%.256e\t', wt(j, i));
		endfor
		fprintf(output_file, '\n');
	endfor
	fclose(output_file);
endfunction

function print_task_init(a, l, t, hx, ht)
	global h0 h1 vareps n k enable_interpolation;

	printf('a = %f\n', a);
	printf('l = %f\n', l);
	printf('T = %f\n', t);
	printf('\n');

	disp('ε =');
	disp(vareps);
	printf('\n');
 
	disp('Параметры сетки:');
	printf('n = %d\n', n);
	printf('k = %d\n', k);
	disp('');

	printf('hx = %f\n', hx);
	printf('ht = %f\n', ht);
	disp('');

	if (enable_interpolation)
		global eta = power(10, -3);
		global xi = power(10, -4);

		disp('Шаги дифференцирования/интегрирования:');
		printf('η = %e\n', eta);
		printf('ξ = %e\n', xi);
		disp('');
	endif

	disp('Начальное отклонение =');
	disp(h0);
	disp('');

	disp('Начальная скорость =');
	disp(h1);
	disp('');
endfunction

function u = osc_u(hx)
	global h0 n k;

	u = zeros(k + 1, n + 1);
	for j = 1 : n + 1
		u(1, j) = h0((j - 1) * hx);
	endfor
endfunction

function res = int_h1_from_zero_to(nth, hx)
	global h1;

	h1_arr = zeros(nth + 1, 1);
	for j = 0 : nth
		eta = j * hx;
		h1_arr(j + 1) = h1(eta);
	endfor

	res = trapz(h1_arr);
endfunction

function v = osc_v(a, l, hx)
	global h1 n k;

	v = zeros(k + 1, n + 1);
	l_arr = zeros(n + 1, 1);
	for nth = 0 : n
		l_arr(nth + 1) = int_h1_from_zero_to(nth, hx);
	endfor
	l_int = trapz(l_arr);

	for j = 1 : n + 1
		x = (j - 1) * hx;
		x_arr = zeros(j, 1);
		for nth = 0 : j - 1
			x_arr(nth + 1) = int_h1_from_zero_to(nth, hx);
		endfor

		v(1, j) = trapz(x_arr) / a - hx * l_int / (a * l);
	endfor

	for i = 1 : k + 1
		v(i, 1) = 0.0;
		v(i, n + 1) = 0.0;
	endfor

	%disp(v(1, :));
endfunction

function print_dampers_info(wt)
	global num_of_dampers k damper_x wt_bounds;
	printf('Всего демпферов: %d\n', num_of_dampers);
	printf('\n');
	
	for i = 1 : num_of_dampers
		printf('Демпфер # %d\n\n', i);

		printf('x_%d = %f\n\n', i, damper_x(i));
		
		printf('Верхнее предельное значение w_%d(t) =\n', i);
		disp(wt_bounds(i, 1));
		printf('\n');
		
		printf('Нижнее предельное значение w_%d(t) =\n', i);
		disp(wt_bounds(i, 2));
		printf('\n');
	endfor
endfunction

function plot_grid_black_and_white(err, a, l, t, hx, ht, y, density, dim, print_automatically)

	global is_empirical task n k xr yr num_of_dampers;

	plot_x = density + 1;
	plot_y = density + 1;
	plot_hx = l / (plot_x - 1);
	plot_ht = t / (plot_y - 1);
	
	xr = 0.0 : hx : l;
	yr = 0.0 : ht : t;

	meshX = linspace(0, l, plot_x);
	meshY = linspace(0, t, plot_y);
	meshZ = zeros(plot_y, plot_x);

	for i = 1 : plot_y
		meshZ(i, :) = interp2(xr, yr, y, 0.0 : plot_hx : l, (i - 1) * plot_ht, 'spline');
	end

	figure;
	mesh(meshX, meshY, meshZ);
 	legend(sprintf('E(T) = %e   ', err));
	xlabel 'X-axis';
	ylabel 'T-axis';
	zlabel 'U-axis';
	title ('U(x, t)');

	if (print_automatically)
		print(sprintf('tasks/task%d_%ddampers_t_%f_grid.eps', task, num_of_dampers, t), dim);
	else
		pause;
	endif
endfunction

function plot_grid(err, a, l, t, hx, ht, y, density, dim, print_automatically)

	global is_empirical task n k xr yr num_of_dampers;

	plot_x = density + 1;
	plot_y = density + 1;
	plot_hx = l / (plot_x - 1);
	plot_ht = t / (plot_y - 1);
	
	xr = 0.0 : hx : l;
	yr = 0.0 : ht : t;

	meshX = linspace(0, l, plot_x);
	meshY = linspace(0, t, plot_y);
	meshZ = zeros(plot_y, plot_x);

	for i = 1 : plot_y
		meshZ(i, :) = interp2(xr, yr, y, 0.0 : plot_hx : l, (i - 1) * plot_ht, 'spline');
	end

	figure;
	mesh(meshX, meshY, meshZ);
 	legend(sprintf('E(T) = %e   ', err));
	xlabel 'X-axis';
	ylabel 'T-axis';
	zlabel 'U-axis';
	title ('U(x, t)');

	if (print_automatically)
		print(sprintf('tasks/task%d_%ddampers_t_%f_grid.eps', task, num_of_dampers, t), '-color', dim);
	else
		pause;
	endif
endfunction

function plot_wt_black_and_white(wt, wt_bounds, t, dim, print_to_file)
	nth = 1;
	global task k num_of_dampers is_empirical;

	xf = [0 : power(10, -6) : t];
	%xf = [0 : t / k : t];
	xp = [0 : t / k : t];

	wt_nth_a = zeros(k + 1, 1);
	wt_nth_b = zeros(k + 1, 1);
	wt_nth_c = zeros(k + 1, 1);
	wt_nth_d = zeros(k + 1, 1);

	wt_size = size(wt);
	for i = 1 : k + 1
		wt_nth_a(i) = wt(i, 1);

		if (wt_size(2) > 1)
			wt_nth_b(i) = wt(i, 2);
		endif

		if (wt_size(2) > 2)
			wt_nth_c(i) = wt(i, 3);
		endif

		if (wt_size(2) > 3)
			wt_nth_d(i) = wt(i, 4);
		endif		
	end

	spl = interp1(xp, wt_nth_a, xf, 'spline');
	spl_d = interp1(xp, wt_nth_a, xp, 'spline');
	spl_b = interp1(xp, wt_nth_b, xf, 'spline');
	spl_d_b = interp1(xp, wt_nth_b, xp, 'spline');
	spl_c = interp1(xp, wt_nth_c, xf, 'spline');
	spl_d_c = interp1(xp, wt_nth_c, xp, 'spline');
	spl__d = interp1(xp, wt_nth_d, xf, 'spline');
	spl_d_d = interp1(xp, wt_nth_d, xp, 'spline');

	is_constrained = false;
	if (!isnan(wt_bounds(1, 1)))
		is_constrained = true;
		ulxf = [0 : power(10, -4) : t];
		ulxp = [0 : t / 20 : t];
		ul_nth = ones(21, 1);
		for i = 1 : 21
			ul_nth(i) = wt_bounds(1, 1);
		end
		ul_d = interp1(ulxp, ul_nth, ulxp, 'spline');
	endif

	if (!isnan(wt_bounds(1, 2)))
		is_constrained = true;
		llxf = [0 : power(10, -4) : t];
		llxp = [0 : t / 20 : t];
		ll_nth = ones(21, 1);
		for i = 1 : 21
			ll_nth(i) = wt_bounds(1, 2);
		end
		ll_d = interp1(llxp, ll_nth, llxp, 'spline');
	endif

	figure;
	cmap = hsv(4);
	if (is_constrained)
		if (wt_size(2) == 1)
			plot(xf, spl, '-*3;;k', ulxp, ul_d, 'LineWidth', 5, 'r', llxp, ll_d, 'LineWidth', 5, 'r');
		elseif (wt_size(2) == 2)
			plot(xf, spl, '-*3;;k', xp, spl_d_b, '-d3;;k', ulxp, ul_d, 'LineWidth', 5, 'r', llxp, ll_d, 'LineWidth', 5, 'r');
		elseif (wt_size(2) == 4)
			plot(xf, spl, 'k', xf, spl_b, '--3;;k', xp, spl_d_c, '-*3;;k', xp, spl_d_d, '-d3;;k', ulxp, ul_d, 'LineWidth', 5, 'r', llxp, ll_d, 'LineWidth', 5, 'r');
		endif
	else
		if (wt_size(2) == 1)
			plot(xf, spl, 'k', xp, spl_d, '*3;;k');
			%plot(xf, spl, '-*3;;k');
		elseif (wt_size(2) == 2)
			%plot(xf, spl, 'k', xf, spl_b, '--3;;k');
			plot(xf, spl, '-*3;;k', xp, spl_d_b, '-d3;;k');
		elseif (wt_size(2) == 4)
			plot(xf, spl, 'k', xf, spl_b, '--3;;k', xp, spl_d_c, '-*3;;k', xp, spl_d_d, '-d;;k');
		endif
	endif
	
	grid on;
	if (wt_size(2) == 1)
		legend('W_1(t)');
	elseif (wt_size(2) == 2)
		if (is_empirical)
			legend('W_1(t)EMP  ', 'W_2(t)EMP  ');
		else
			legend('W_1(t)', 'W_2(t)');
		endif
	elseif (wt_size(2) == 4)
		legend('W_1(t)', 'W_2(t)', 'W_3(t)', 'W_4(t)');
	endif
	xlabel 't';
	ylabel 'W(t)';
	title 'W(t)';

	if (print_to_file)
		if (is_empirical)
			print(sprintf('tasks/task%d_%ddampers_t_%f_empirical_wt.eps', task, num_of_dampers, t), dim);
		else
			print(sprintf('tasks/task%d_%ddampers_t_%f_wt.eps', task, num_of_dampers, t), dim);
		endif
	else
		pause;
	endif
endfunction

function plot_wt(wt, wt_bounds, t, dim, print_to_file)
	nth = 1;
	global task k num_of_dampers is_empirical;

	xf = [0 : power(10, -6) : t];
	%xf = [0 : t / k : t];
	xp = [0 : t / k : t];

	wt_nth_a = zeros(k + 1, 1);
	wt_nth_b = zeros(k + 1, 1);
	wt_nth_c = zeros(k + 1, 1);
	wt_nth_d = zeros(k + 1, 1);

	wt_size = size(wt);
	for i = 1 : k + 1
		wt_nth_a(i) = wt(i, 1);

		if (wt_size(2) > 1)
			wt_nth_b(i) = wt(i, 2);
		endif

		if (wt_size(2) > 2)
			wt_nth_c(i) = wt(i, 3);
		endif

		if (wt_size(2) > 3)
			wt_nth_d(i) = wt(i, 4);
		endif		
	end

	spl = interp1(xp, wt_nth_a, xf, 'spline');
	spl_d = interp1(xp, wt_nth_a, xp, 'spline');
	spl_b = interp1(xp, wt_nth_b, xf, 'spline');
	spl_d_b = interp1(xp, wt_nth_b, xp, 'spline');
	spl_c = interp1(xp, wt_nth_c, xf, 'spline');
	spl_d_c = interp1(xp, wt_nth_c, xp, 'spline');
	spl__d = interp1(xp, wt_nth_d, xf, 'spline');
	spl_d_d = interp1(xp, wt_nth_d, xp, 'spline');

	is_constrained = false;
	if (!isnan(wt_bounds(1, 1)))
		is_constrained = true;
		ulxf = [0 : power(10, -4) : t];
		ulxp = [0 : t / 20 : t];
		ul_nth = ones(21, 1);
		for i = 1 : 21
			ul_nth(i) = wt_bounds(1, 1);
		end
		ul_d = interp1(ulxp, ul_nth, ulxp, 'spline');
	endif

	if (!isnan(wt_bounds(1, 2)))
		is_constrained = true;
		llxf = [0 : power(10, -4) : t];
		llxp = [0 : t / 20 : t];
		ll_nth = ones(21, 1);
		for i = 1 : 21
			ll_nth(i) = wt_bounds(1, 2);
		end
		ll_d = interp1(llxp, ll_nth, llxp, 'spline');
	endif

	figure;
	cmap = hsv(4);
	if (is_constrained)
		if (wt_size(2) == 1)
			plot(xf, spl, 'm', xp, spl_d, '*3;;b', ulxp, ul_d, 'LineWidth', 5, 'r', llxp, ll_d, 'LineWidth', 5, 'r');
		elseif (wt_size(2) == 2)
			plot(xf, spl, 'm', xf, spl_b, 'g', xp, spl_d, '*3;;b', xp, spl_d_b, '*3;;b', ulxp, ul_d, 'LineWidth', 5, 'r', llxp, ll_d, 'LineWidth', 5, 'r');
		elseif (wt_size(2) == 4)
			plot(xf, spl, 'Color', cmap(1, :), xf, spl_b, 'Color', cmap(2, :), xf, spl_c, 'Color', cmap(3, :), xf, spl__d, 'Color', cmap(4, :), xp, spl_d, '*3;;b', xp, spl_d_b, '*3;;b', ulxp, ul_d, 'LineWidth', 5, 'r', llxp, ll_d, 'LineWidth', 5, 'r');
		endif
	else
		if (wt_size(2) == 1)
			plot(xf, spl, 'm', xp, spl_d, '*3;;b');
		elseif (wt_size(2) == 2)
			if (is_empirical)
				plot(xf, spl, 'Color', cmap(1, :), xf, spl_b, 'Color', cmap(2, :));
			else
				plot(xf, spl, 'Color', cmap(3, :), xf, spl_b, 'Color', cmap(4, :));
			endif
		elseif (wt_size(2) == 4)
			plot(xf, spl, 'Color', cmap(1, :), xf, spl_b, 'Color', cmap(2, :), xf, spl_c, 'Color', cmap(3, :), xf, spl__d, 'Color', cmap(4, :), xp, spl_d, '*3;;b', xp, spl_d_b, '*3;;b', xp, spl_d_c, '*3;;b', xp, spl_d_d, '*3;;b');
		endif
	endif
	
	grid on;
	if (wt_size(2) == 1)
		legend('W_1(t)');
	elseif (wt_size(2) == 2)
		if (is_empirical)
			legend('W_1(t)EMP  ', 'W_2(t)EMP  ');
		else
			legend('W_1(t)', 'W_2(t)');
		endif
	elseif (wt_size(2) == 4)
		legend('W_1(t)', 'W_2(t)', 'W_3(t)', 'W_4(t)');
	endif
	xlabel 't';
	ylabel 'W(t)';
	title 'W(t)';

	if (print_to_file)
		if (is_empirical)
			print(sprintf('tasks/task%d_%ddampers_t_%f_empirical_wt.eps', task, num_of_dampers, t), '-color', dim);
		else
			print(sprintf('tasks/task%d_%ddampers_t_%f_wt.eps', task, num_of_dampers, t), '-color', dim);
		endif
	else
		pause;
	endif
endfunction