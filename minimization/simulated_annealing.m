1;

global boltzmann_a = 0.0;
global boltzmann_b = 0.0;
global boltzmann_c = 0.0;

global boltzmann_x = 0.0;
global boltzmann_y = 0.0;

function t = boltzmann_annealing(t0, k)
	t = t0 / log(1 + k);
endfunction

function h = new_random_step(t)
	a = -1000;
	b = 1000;
	h = a + (b - a) * rand(1);
endfunction

function wt = new_random_move(wt, t)	
	for i = 1 : length(wt(:))
	
		wt(i) = wt(i) + new_random_step(t);
	endfor
endfunction

function h = random_step(t)
	a = -t;
	b = t;
	h = a + (b - a) * rand(1);
endfunction

function wt = random_move(wt, t)	
	for i = 1 : length(wt(:))
	
		wt(i) = wt(i) + random_step(t);
	endfor
endfunction

function p = p_gibbs_measure(energy_delta, t)
	p = 1 / (1 + exp(energy_delta / t));
endfunction

function l = wt_length(wt)
	l = 0;
	for i = 1 : length(wt(:))
		l = l + power(wt(i), 2);
	endfor
	l = sqrt(l);
endfunction

function g = boltzmann_density()
	global t boltzmann_x boltzmann_y;
	g = power(2 * t * pi, -1 / 2) * exp(-(power(boltzmann_x, 2) + power(boltzmann_y, 2)) / (2 * t));
endfunction

function ix = boltzmann_sub_x(xi)
	global boltzmann_a boltzmann_x;
	boltzmann_x = boltzmann_a - (1 - xi) / xi;

	ix = boltzmann_density() / power(xi, 2);
endfunction

function p = boltzmann_int_x()
	p = quad(@boltzmann_sub_x, 0, 1);
	
	#TODO	Fix quad integration!!!
	if (abs(p) <= power(10, -8))
		p = 1.0;
	endif
endfunction

function iy = boltzmann_sub_y(eta)
	global boltzmann_b boltzmann_y;
	boltzmann_y = boltzmann_b - (1 - eta) / eta;
	
	iy = boltzmann_int_x() / power(eta, 2);
endfunction

function p = boltzmann_int_y()
	p = quad(@boltzmann_sub_y, 0, 1);
	
	#TODO	Fix quad integration!!!
	if (abs(p) <= power(10, -8))
		p = 1.0;
	endif
	
	#p = boltzmann_int_x();
endfunction

function p = boltzmann_p(wt, wt_new)
	global t boltzmann_a boltzmann_b boltzmann_c;
	
	d = length(wt(:));
	boltzmann_end = zeros(1, d);
	for i = 1 : d
		boltzmann_end(i) = abs(wt_new(i) - wt(i));
	endfor

	p = power(2 * pi * t, -d / 2) * (power(pi, d / 2) * power(t, d / 2) / power(2, d / 2));
	for i = 1 : d
		p = p * (1 + erf(boltzmann_end(i) / sqrt(2 * t)));
	endfor

endfunction

function h = random_value_in_ranges(a, b)
	h = a + (b - a) * rand(1);
endfunction

function wt = new_new_random_move(wt_prev, t, start_from, end_at)
	wt = wt_prev;
	
	for i = 1 : length(wt(:))
		#wt(i) = wt_prev(i) + random_value_in_ranges(-100, 100);
		#if (wt(i) < -1000)
		#	wt(i) = -1000;
		#elseif (wt(i) > 1000)
		#	wt(i) = 1000;
		#endif

		wt(i) = random_value_in_ranges(start_from, end_at);
	endfor
endfunction

function wt = simulated_annealing(damper_x, wt_start, t0, m, start_from, end_at)

	global t boltzmann_a;

	printf("\nUsing simulated annealing minimization\n\n");
	
	t = t0;
	k = 1;
	
	energy_start = solve_all(damper_x, wt_start);
	gl_min = energy_start;
	
	wt = wt_start;
	wt_global = wt_start;
	
	iter = 1;
	
	
	
	
	#for i = 1 : 30
	#	wt_new = new_new_random_move(wt, t);
	#	disp(wt_new);
	#	printf("\n");
	#endfor
	#pause;
	
	
	
	
	
	
	#disp(int_var_sub());
	
	#m = -1;
	
	#{
	for i = 1 : 1
	
	wt_test_1 = wt_start;
	disp(wt_test_1);
	printf("\n");
	
	wt_test_2 = wt_start;
	wt_test_2(1) = (rand(1) - 0.5);
	disp(wt_test_2);
	printf("\n");
	
	
	
	disp(boltzmann_p(wt_test_1, wt_test_2));
	#disp(boltzmann_a);
	
	re = 1 / 2 * (erf(boltzmann_a / sqrt(2 * t)) + 1);
	disp(re);
	
	endfor
	
	pause;
	#}
	
	#disp('P	1000');
	#disp(boltzmann_p(wt_test_1, wt_test_2, 10));
	#printf("\n");
	
	#endfor
	
	#disp(boltzmann_density(wt_test_1, wt_test_2, 1));
	#printf("\n");
	
	#disp(boltzmann_density(wt_test_1, wt_test_2, 10));
	#printf("\n");
	
	#disp(boltzmann_density(wt_test_1, wt_test_2, 100));
	#printf("\n");
	
	#disp(boltzmann_density(wt_test_1, wt_test_2, 1000));
	#printf("\n");
	
	
	#disp('P	100');
	#disp(boltzmann_p(wt_test_1, wt_test_2, 100));
	#printf("\n");
	
	#disp('P	10');
	#disp(boltzmann_p(wt_test_1, wt_test_2, 10));
	#printf("\n");
	
	#disp('P	1');
	#disp(boltzmann_p(wt_test_1, wt_test_2, 1));
	#printf("\n");
	
	#disp('!!!TEST END!!!');
	
	do
		
		do
		
			wt_alpha = 1;
			wt_p = 0;
			do
				wt_new = new_new_random_move(wt, t, start_from, end_at);
			
				wt_p = 1;	#boltzmann_p(wt, wt_new);
				#printf("Iteration %d/%d new P:\n", iter, m);
				#disp(wt_p);
				#printf("\n");
				
				wt_alpha = rand(1);
				#printf("Iteration %d/%d new Alpha:\n", iter, m);
				#disp(wt_alpha);
				#printf("\n");
			until (wt_p >= wt_alpha)
	
			energy_new = solve_all(damper_x, wt_new);
			if (energy_new < gl_min)
				gl_min = energy_new;
				wt_global = wt_new;
			endif
			
			printf("Iteration %d/%d global minimum w(t):\n", iter, m);
			disp(wt_global);
			printf("\n");
			
			printf("Iteration %d/%d w(t):\n", iter, m);
			disp(wt);
			printf("\n");
			
			printf("Iteration %d/%d possible w(t):\n", iter, m);
			disp(wt_new);
			printf("\n");

			printf("Iteration %d/%d T:\n", iter, m);
			disp(t);
			printf("\n");
			
			printf("Iteration %d/%d global minimum:\n", iter, m);
			disp(gl_min);
			printf("\n");

			printf("Iteration %d/%d start energy:\n", iter, m);
			disp(energy_start);
			printf("\n");
			
			printf("Iteration %d/%d new energy:\n", iter, m);
			disp(energy_new);
			printf("\n");

			energy_delta = energy_new - energy_start;
		
			if (energy_delta >= 0)
				alpha = rand(1);
				printf("Iteration %d/%d alpha:\n", iter, m);
				disp(alpha);
				printf("\n");
	
				p = p_gibbs_measure(energy_delta, t);
	
				printf("Iteration %d/%d P:\n", iter, m);
				disp(p);
				printf("\n");
			else
				alpha = 0;
				p = 1;
			endif

		until (alpha < p)
	
		iter = iter + 1;
		wt = wt_new;
		energy_start = energy_new;
		k = k + 1;
		t = boltzmann_annealing(t0, k);
	until (iter > m)
	
	wt = wt_global;

endfunction
