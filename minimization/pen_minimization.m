1;

global pen_fun;
global pen_limits;
global pen_r;
global pen_c;
global pen_m;
global pen_p;

function g = pm_g(wt, nth)
	global k pen_limits num_of_dampers;

	wt_pm = zeros(k + 1, num_of_dampers);
	
	for i = 1 : k + 1
		for j = 1 : num_of_dampers
			wt_pm(i, j) = pen_limits(j);
		end
	end
	
	#disp(wt_pm);

	g = max(0, wt(nth) - wt_pm(nth));
	#disp(g);
endfunction

function g = pm_sum_one(damper_x)
	g = 0;
endfunction

function g = pm_sum_two(wt)
	global num_of_dampers k wt_bounds;

	g = 0;
	for i = 1 : num_of_dampers
		if (wt_bounds(i, 1) != nan)
			for j = 1 : k + 1
				pen = max(0, wt(j, i) - wt_bounds(i, 1));
				g = g + power(pen, 2);
			endfor
		endif
		if (wt_bounds(i, 2) != nan)
			for j = 1 : k + 1
				pen = max(0, wt_bounds(i, 2) - wt(j, i));
				g = g + power(pen, 2);
			endfor
		endif
	endfor
endfunction

function p = pen(damper_x, wt)
	global pen_r;

	p = pen_r * (pm_sum_one(damper_x) + pm_sum_two(wt)) / 2;
endfunction

function f = pm_f(damper_x, wt)
	global pen_fun pen_limits pen_r pen_m pen_p;

	#disp(penalty);

	f = pen_fun(damper_x, wt) + pen(damper_x, wt);
endfunction

function wt = pen_minimization(fun, damper_x, wt_bounds, wt_start, vareps, r, c)
	global pen_fun pen_limits pen_r pen_c pen_m pen_p vareps;

	printf("\nUsing penalty method minimization\n\n");
	
	pen_fun = fun;
	k = 0;
	m = 0;
	#disp(p);
	
	wt = wt_start;
	pen_limits = wt_bounds;
	pen_r = r;
	pen_c = c;
	pen_m = m;
	pen_p = length(wt_start(:)) * 2;
	#pm_f(damper_x, wt_prev, wt_bounds, r, m, p);
	#wt = pm_hooke_jeeves(damper_x, wt_prev, wt_bounds, 1, 1, 0, 2, r, m, p);
	#wt = marquardt(damper_x, wt_prev, 0.1);
	
	do
		wt = marquardt(@pm_f, damper_x, wt, inf, vareps);
		
		p = pen(damper_x, wt);
		pen_r = pen_c * pen_r;
		
	until (p <= vareps)
	
endfunction
