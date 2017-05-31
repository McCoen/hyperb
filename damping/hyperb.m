1;

global spl_x;

function print_task_init(t, hx, ht)
	global boundary a l vareps n k enable_interpolation;

	printf("a = %f\n", a);
	printf("l = %f\n", l);
	printf("T = %f\n", t);
	printf("\n");

	disp('ε =');
	disp(vareps);
	printf("\n");
 
	disp('Параметры сетки:');
	printf("n = %d\n", n);
	printf("k = %d\n", k);
	disp('');

	printf("hx = %f\n", hx);
	printf("ht = %f\n", ht);
	disp('');

	if (enable_interpolation)
		global eta = power(10, -3);
		global xi = power(10, -4);

		disp('Шаги дифференцирования/интегрирования:');
		printf("η = %e\n", eta);
		printf("ξ = %e\n", xi);
		disp('');
	endif

	disp('Начальное возмущение =');
	disp(boundary);
	disp('');
endfunction

function alpha = osc_alpha(hx, ht)
	global a;
	alpha = 2 * power(hx, 2) / (a * ht);
endfunction

function beta = osc_beta(hx)
	global a;
	beta = power(hx, 2) / a;
endfunction

function b = osc_b()
	b = [0, -1; 1, 0];
endfunction

function c = osc_c(alpha, b)
	c = 2 * eye(2) + alpha * b;
endfunction

function cTilde = osc_cTilde(alpha, b)
	cTilde = 2 * eye(2) - alpha * b;
endfunction

function y = osc_y(hx)
	global boundary n k;
	y = zeros(k + 1, n + 1);
	for i = 1 : k + 1
		for j = 1 : n + 1
			if (i == 1)
				y(i, j) = boundary((j - 1) * hx);
			endif
		end
	end
endfunction

function y_sec = osc_sec_y()
	global n k;
	y_sec = zeros(k + 1, n + 1);
endfunction

function res = solve_all(wt, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)
	global n k damper_x;
	
	
	y = oscDeadening(damper_x, wt, n, k, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	res = energyInt(damper_x, wt, n, k, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	
endfunction

function print_dampers_info()
	global num_of_dampers k damper_x wt_bounds wt;
	printf("Всего демпферов: %d\n", num_of_dampers);
	printf("\n");
	
	for i = 1 : num_of_dampers
		printf("Демпфер # %d\n\n", i);

		printf("x_%d = %f\n\n", i, damper_x(i));
		
		printf("Верхнее предельное значение w_%d(t) =\n", i);
		disp(wt_bounds(i, 1));
		printf("\n");
		
		printf("Нижнее предельное значение w_%d(t) =\n", i);
		disp(wt_bounds(i, 2));
		printf("\n");

		printf("Управляющая функция w_%d(t):\n", i);
		disp(wt(:, i));
		printf("\n");
	end
endfunction

function extended_print_grid_to_file(u, n, k, hx, ht, filename)
	output_file = fopen(filename, 'w');
	
	fprintf(output_file, "hx = %f\n", hx);
	fprintf(output_file, "ht = %f\n\n", ht);
	
	fprintf(output_file, "u = [");

	for i = 1 : k
		for j = 1 : n
			fprintf(output_file, "%.32e, ", u(i, j));
		endfor
		fprintf(output_file, "%.32e;\n", u(i, n + 1));
	endfor

	for j = 1 : n
		fprintf(output_file, "%.32e, ", u(k + 1, j));
	endfor
	fprintf(output_file, "%.32e];\n", u(k + 1, n + 1));

	fclose(output_file);
endfunction

function print_u_grid_values(ht, y)
	global n k;
	
	disp('u(x, t)');
	for i = 1 : k + 1
		printf("T = %f: ", ht * (i - 1));
		for j = 1 : n + 1
			printf("%f ", y(i, j));
		end
		printf("\n");
	end
endfunction

function plot_u_cut()

	global task hx ht n k l t_total y xr yr;

	plot_x = 76;
	plot_y = 76;
	plot_hx = l / (plot_x - 1);
	plot_ht = t_total / (plot_y - 1);

	meshX = linspace(0, l, plot_x);
	meshY = linspace(0, t_total, plot_y);
	meshZ = zeros(plot_y, plot_x);

	for i = 1 : plot_y
		#for j = 1 : plot_x
		#	meshZ(i, j) = interp2(xr, yr, y, (j - 1) * plot_hx, (i - 1) * plot_ht, "spline");
		#end
		meshZ(i, :) = interp2(xr, yr, y, 0.0 : plot_hx : l, (i - 1) * plot_ht, "spline");
	end

	mesh(meshX, meshY, meshZ);
	legend(sprintf("E(T) = %e", err));
	xlabel 'X-axis';
	ylabel 'T-axis';
	zlabel 'U-axis';
	title ('U(x, t)');
	print(sprintf("images/task %d u(x, t) t %f.png", task, t_total), "-S1280, 720");
endfunction

function plot_u(u)

	global l t_total;
	qn = length(u);
	xf = cell(1, qn);
	xp = cell(1, qn);
	spl = cell(1, qn);
	spl_d = cell(1, qn);
	
	for nth = 1 : qn
		dim = size(cell2mat(u(nth)));
		
		k = dim(1) - 1;
		n = dim(2) - 1;
		hx = l / n;
		ht = t_total / k;
		
		xr = 0.0 : hx : l;
		yr = 0.0 : ht : t_total;
		
		xf(nth) = [0 : power(10, -6) : t_total];
		xp(nth) = [0 : t_total / k : t_total];

		u_current = cell2mat(u(nth));
		damper_x = 0.75;
		spl_x = interp2(xr, yr, u_current, damper_x, 0.0 : ht : t_total, "spline");
		spl(nth) = interp1(cell2mat(xp(nth)), spl_x, cell2mat(xf(nth)), "spline");
		spl_d(nth) = interp1(cell2mat(xp(nth)), spl_x, cell2mat(xp(nth)), "spline");
	endfor

	cmap = hsv(qn);
	plot(cell2mat(xf(1)), cell2mat(spl(1)), 'Color', cmap(1, :),
		cell2mat(xf(2)), cell2mat(spl(2)), 'Color', cmap(2, :),
		cell2mat(xf(3)), cell2mat(spl(3)), 'Color', cmap(3, :),
		cell2mat(xf(4)), cell2mat(spl(4)), 'Color', cmap(4, :),
		cell2mat(xf(5)), cell2mat(spl(5)), 'Color', cmap(5, :),
		cell2mat(xf(6)), cell2mat(spl(6)), 'Color', cmap(6, :));

	grid on;
	legend(sprintf("8 x 16"), sprintf("12 x 24"), sprintf("16 x 32"), sprintf("24 x 48"), sprintf("32 x 64"), sprintf("48 x 96"));
	xlabel 't';
	ylabel(sprintf("U(%.2f, t)", damper_x));
	title(sprintf("U(x = %.2f, t = %.2f)", damper_x, t_total));
	print(sprintf("images/task %d u(x, t) cut at %f time %f.png", 2, damper_x, t_total), "-S1280, 720");
endfunction

function plot_u_grid(err, t, hx, ht, y)

	global task n k l xr yr;

	plot_x = 101;
	plot_y = 101;
	plot_hx = l / (plot_x - 1);
	plot_ht = t / (plot_y - 1);
	
	xr = 0.0 : hx : l;
	yr = 0.0 : ht : t;

	meshX = linspace(0, l, plot_x);
	meshY = linspace(0, t, plot_y);
	meshZ = zeros(plot_y, plot_x);

	for i = 1 : plot_y
		#for j = 1 : plot_x
		#	meshZ(i, j) = interp2(xr, yr, y, (j - 1) * plot_hx, (i - 1) * plot_ht, "spline");
		#end
		meshZ(i, :) = interp2(xr, yr, y, 0.0 : plot_hx : l, (i - 1) * plot_ht, "spline");
	end

	mesh(meshX, meshY, meshZ);
# 	legend(sprintf("E(T) = %e", err));
	xlabel 'X-axis';
	ylabel 'T-axis';
	zlabel 'U-axis';
	title ('U(x, t)');
%  	print(sprintf("images/task %d u(x, t) t %f.png", task, t), "-S800, 600");
  	print(sprintf("images/task %d u(x, t) t %f.png", task, t), "-S1366, 1024");
endfunction

function plot_f(damper_x, wt)
	global task n num_of_dampers wt_limits l;

	xf = [0 : power(10, -4) : l];
	xp = [0 : l / n : l];

	f_nth_1 = ones(n + 1, 1);
	f_nth_2 = ones(n + 1, 1);
	f_nth_3 = ones(n + 1, 1);
	f_nth_4 = ones(n + 1, 1);
	f_nth_5 = ones(n + 1, 1);
	f_nth_6 = ones(n + 1, 1);
	f_nth_7 = ones(n + 1, 1);
	f_nth_8 = ones(n + 1, 1);
	f_nth_9 = ones(n + 1, 1);
	f_nth_10 = ones(n + 1, 1);
	for i = 1 : n + 1
		f_nth_1(i) = f(1, i, damper_x, wt);
		f_nth_2(i) = f(2, i, damper_x, wt);
		f_nth_3(i) = f(3, i, damper_x, wt);
		f_nth_4(i) = f(4, i, damper_x, wt);
		f_nth_5(i) = f(5, i, damper_x, wt);
		f_nth_6(i) = f(6, i, damper_x, wt);
		f_nth_7(i) = f(7, i, damper_x, wt);
		f_nth_8(i) = f(8, i, damper_x, wt);
		f_nth_9(i) = f(9, i, damper_x, wt);
		f_nth_10(i) = f(10, i, damper_x, wt);
	end
	
	spl_1 = interp1(xp, f_nth_1, xf, "linear");
	spl_2 = interp1(xp, f_nth_2, xf, "linear");
	spl_3 = interp1(xp, f_nth_3, xf, "linear");
	spl_4 = interp1(xp, f_nth_4, xf, "linear");
	spl_5 = interp1(xp, f_nth_5, xf, "linear");
	spl_6 = interp1(xp, f_nth_6, xf, "linear");
	spl_7 = interp1(xp, f_nth_7, xf, "linear");
	spl_8 = interp1(xp, f_nth_8, xf, "linear");
	spl_9 = interp1(xp, f_nth_9, xf, "linear");
	spl_10 = interp1(xp, f_nth_10, xf, "linear");
	
	cmap = hsv(10);
	plot(xf, spl_1, 'Color', cmap(1, :),
		xf, spl_2, 'Color', cmap(2, :),
		xf, spl_3, 'Color', cmap(3, :),
		xf, spl_4, 'Color', cmap(4, :),
		xf, spl_5, 'Color', cmap(5, :),
		xf, spl_6, 'Color', cmap(6, :),
		xf, spl_7, 'Color', cmap(7, :),
		xf, spl_8, 'Color', cmap(8, :),
		xf, spl_9, 'Color', cmap(9, :),
		xf, spl_10, 'Color', cmap(10, :));
	grid on;
	xlabel 't';
	ylabel 'f';
	title ('f(x, t)');
	print(sprintf("images/task %d f %d damper(s).png", task, num_of_dampers), "-S1280, 720");
	#print(strcat('task ', int2str(task), ' w(t) damper ', int2str(nth), ' of ', int2str(num_of_dampers), '.png'), "-S640, 480");
	#print(strcat('task ', int2str(task), ' w(t) damper ', int2str(nth), ' of ', int2str(num_of_dampers), '.png'), "-S640, 480");
endfunction

function plot_wt(wt, t, nth)
	global task k num_of_dampers wt_bounds;

	xf = [0 : power(10, -6) : t];
	xp = [0 : t / k : t];
	wt_nth = ones(k + 1, 1);
	for i = 1 : k + 1
		wt_nth(i) = wt(i, nth);
	end
	spl = interp1(xp, wt_nth, xf, "spline");
	spl_d = interp1(xp, wt_nth, xp, "spline");
	
	#{
	ulxf = [0 : power(10, -4) : 0.2];
	ulxp = [0 : 0.01 : 0.2];
	ul_nth = ones(21, 1);
	for i = 1 : 21
		ul_nth(i) = wt_bounds(nth);
	end
	ul_d = interp1(ulxp, ul_nth, ulxp, "spline");
	#}
	
	#plot(xf, spl, "m", xp, spl_d, "+3;;b", ulxp, ul_d, "r");
	#plot(xf, spl, "m", xp, spl_d, "+3;;b");
	plot(xf, spl, "m", xp, spl_d, "+3;;b");
	grid on;
# 	legend(sprintf("123"));
	xlabel 't';
	ylabel 'W(t)';
	title (strcat('W', '_', int2str(nth), '(t)'));
	print(sprintf("images/task %d w(t) damper %d of %d.png", task, nth, num_of_dampers), "-S1280, 720");
endfunction

function plot_start_h()
	global task hx k num_of_dampers;

	xf = [0 : power(10, -4) : 1.0];
	xp = [0 : 0.01 : 1.0];
	wt_nth = ones(101, 1);
	for x = 1 : 101
		wt_nth(x) = boundary((x - 1) * 0.01);
	end
	spl = interp1(xp, wt_nth, xf, "spline");
	
	df = [0 : power(10, -4) : 1.0];
	dp = [0 : boundary(0.5) / 40 : boundary(0.5)];
	d_nth = ones(41, 1);
	
	for x = 1 : 41
		d_nth(x) = 0.5;
	end
	spl_d = interp1(dp, d_nth, dp, "spline");
	
	d2f = [0 : power(10, -4) : 1.0];
	d2p = [0 : boundary(0.5) / 40 : boundary(0.5)];
	d2_nth = ones(41, 1);
	
	for x = 1 : 41
		d2_nth(x) = 0.5;
	end
	spl_d2 = interp1(d2p, d2_nth, d2p, "spline");
	
	d3f = [0 : power(10, -4) : 1.0];
	d3p = [0 : boundary(0.25) / 40 : boundary(0.25)];
	d3_nth = ones(81, 1);
	
	for x = 1 : 81
		d3_nth(x) = 0.75;
	end
# 	spl_d3 = interp1(d3p, d3_nth, d3p, "spline");
	
	#plot(xf, spl, "m", spl_d, dp, "+3;;b", spl_d2, d2p, "+3;;b", spl_d3, d3p, "+3;;b");
	plot(xf, spl, "m", spl_d, dp, "+3;;b");
	grid on;
	xlabel 'x';
	ylabel 'H(x)';
	title ('H_0(x)');
	print(sprintf("images/task %d h_0 %d damper(s).png", task, num_of_dampers), "-S1280, 720");
endfunction
