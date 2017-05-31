1;

function ei = solve_marquardt(damper_x, wt_start, hx, ht)
	global vareps;
	alpha = osc_alpha(hx, ht);
	beta = osc_beta(hx);
	
	b = osc_b();
	c = osc_c(alpha, b);
	c_inverted = inv(c);
	sweep_side = eye(2);
	
	cTilde = osc_cTilde(alpha, b);
	
	y = osc_y(hx);
	y_sec = osc_sec_y();
	
	wt = marquardt(@solve_all, damper_x, wt_start, 100, vareps, power(10, 4), hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta, true);
	ei = power(10, -4) - solve_all(wt, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
endfunction

function t = newton(t_a, t_b, damper_x, wt_start)
	global l n k;
	hx = l / n;
	ht_a = t_a / k;
	ht_b = t_b / k;
	
	t = (t_b + t_a) / 2;
	
	total = 20;
	rep = zeros(total, 1);
	for i = 1 : total
		ht = t / k;
		ei = solve_marquardt(damper_x, wt_start, hx, ht);
		
		t2 = t + power(10, -4);
		ht2 = t2 / k;
		ei2 = solve_marquardt(damper_x, wt_start, hx, ht2);
		
		der = (t2 - t) / power(10, -4);
		
		t = t - ei / der;
		rep(i) = t;
		disp('T = ');
		disp(t);
	endfor
	disp(rep);
	
#  	ei = solve_marquardt(damper_x, wt_start, hx, ht_a);
# 	ei2 = solve_marquardt(damper_x, wt_start, hx, ht_b);
# 	disp('Ei');
#  	disp(ei);
# 	disp(ei2);

%  global xr = 0.0 : hx : l;
%  global yr = 0.0 : ht : t_total;
	t = 0.125;
endfunction