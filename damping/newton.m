1;

function ei = solve_marquardt(damper_x, wt_start, hx, ht, t, nth)
	global vareps wt_bounds st num_of_dampers n k a l;
	alpha = osc_alpha(hx, ht);
	beta = osc_beta(hx);
	
	b = osc_b();
	c = osc_c(alpha, b);
	c_inverted = inv(c);
	sweep_side = eye(2);
	
	cTilde = osc_cTilde(alpha, b);
	
	u = osc_y(hx);
	v = osc_sec_y();
	
	%wt = marquardt(@solve_all, damper_x, wt_start, 100, vareps, power(10, 4), hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta, true);
	mq_max_iterations = inf;
	output_filename = "mqlast";
	wt = gpuMarquardtMinimization(damper_x, wt_bounds, wt_start, st, num_of_dampers, n, k, u, a, l, t, mq_max_iterations, vareps, output_filename);
	
	%ei = power(10, -4) - solve_all(wt, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	current_energy_int = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, a, l, t);
	%last_err(nth) = current_energy_int;
	ei = power(10, -4) - current_energy_int;
endfunction

function t = newton_nonlinear_solver(t_a, t_b, damper_x, wt_start)
	global l n k;
	hx = l / n;
	ht_a = t_a / k;
	ht_b = t_b / k;
	
	t = (t_b + t_a) / 2;
	
	total = 30;
	rep = zeros(total, 1);
	last_err = zeros(total, 1);
	for i = 1 : total
		ht = t / k;
		ei = solve_marquardt(damper_x, wt_start, hx, ht, t, i);
		last_err(i) = ei;
		printf("Newton T = %e ei = %e error = %e\n", t, ei, power(10, -4) - ei);
		%disp('Debug EI');
		%disp(ei);
		
		t2 = t + power(10, -4);
		ht2 = t2 / k;
		ei2 = solve_marquardt(damper_x, wt_start, hx, ht2, t2, i);
		%disp('Debug EI');
		%disp(ei2);
		
		der = (t2 - t) / power(10, -4);
		%disp('Debug der');
		%disp(der);
		
		t = t - ei / der;
		rep(i) = t;
		disp('Newton T = ');
		disp(t);
	endfor

	disp('Total Newton iterations T');
	disp(rep);
	disp('Total Newton iterations error');
	disp(last_err);
	
#  	ei = solve_marquardt(damper_x, wt_start, hx, ht_a);
# 	ei2 = solve_marquardt(damper_x, wt_start, hx, ht_b);
# 	disp('Ei');
#  	disp(ei);
# 	disp(ei2);

%  global xr = 0.0 : hx : l;
%  global yr = 0.0 : ht : t_total;
%	t = 0.125;
endfunction
