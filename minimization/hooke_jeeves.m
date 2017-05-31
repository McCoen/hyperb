1;

function d = hj_d(h, i, j)
	global num_of_dampers k;
	d = zeros(num_of_dampers, k + 1);
	d(i, j) = h;
endfunction

function wt = hooke_jeeves(damper_x, wt_prev, vareps, h, lambda, hj_alpha)
	global k t_total;
	
	printf("\nUsing Hooke-Jeeves minimization\n\n");
	
	iter = 0;
	f_orig = new_new_solve_all(damper_x, wt_prev);
	
	printf("Iteration #%d error: %f\n", iter, f_orig);
	
	while true	
		
		iter = iter + 1;
	
		wt_orig = wt_prev;
		
		f_prev = new_new_solve_all(damper_x, wt_prev);
	
	
		for i = 1 : length(damper_x)
			for j = 1 : k + 1
				wt_new = wt_prev + hj_d(h, i, j);
				
				f_new = new_new_solve_all(damper_x, wt_new);
			
				if (f_new < f_prev)
		
		
					wt_prev = wt_new;	# success
			
			
				elseif (f_new >= f_prev)
		
		
					wt_new = wt_prev - hj_d(h, i, j);
					f_new = new_new_solve_all(damper_x, wt_new);
			
					if (f_new < f_prev)
						wt_prev = wt_new;	# success
					endif
						
				endif
				
			end
			
		end
	
		
		printf("Iteration #%d w(t):\n", iter);
		disp(wt_prev);
		disp(' ');
		printf("Iteration #%d h(t):\n", iter);
		disp(h);
		disp(' ');
	
	
		f_new = new_new_solve_all(damper_x, wt_prev);
	
		do_next = false;
		if (f_new < f_orig)
			#wt_prev = wt_prev + lambda * (wt_prev - wt_orig);
			f_orig = new_new_solve_all(damper_x, wt_prev);
			do_next = true;
		else
			if (h > vareps)
				printf("%f -> ", h);
				h = h / hj_alpha;
				printf("%f\n\n", h);
				do_next = true;
			endif
		endif
				
		printf("Iteration #%d error: %f\n", iter, f_orig);
		disp(' ');
		if (do_next == false)
			break;
		endif
	end
	wt = wt_prev;
endfunction

function wt = pm_hooke_jeeves(damper_x, wt_prev, wt_limits, vareps, h, lambda, hj_alpha, r, m, p)
	global k t_total;
	
	printf("\nUsing Hooke-Jeeves minimization\n\n");
	
	iter = 0;
	#f_orig = new_new_solve_all(damper_x, wt_prev);
	f_orig = pm_f(damper_x, wt_prev, wt_limits, r, m, p);
	
	printf("Iteration #%d error: %f\n", iter, f_orig);
	
	while true	
		
		iter = iter + 1;
	
		wt_orig = wt_prev;
		
		#f_prev = new_new_solve_all(damper_x, wt_prev);
		f_prev = pm_f(damper_x, wt_prev, wt_limits, r, m, p);
	
	
		for i = 1 : length(damper_x)
			for j = 1 : k + 1
				wt_new = wt_prev + hj_d(h, i, j);
				
				#f_new = new_new_solve_all(damper_x, wt_new);
				f_new = pm_f(damper_x, wt_new, wt_limits, r, m, p);
			
				if (f_new < f_prev)
		
		
					wt_prev = wt_new;	# success
			
			
				elseif (f_new >= f_prev)
		
		
					wt_new = wt_prev - hj_d(h, i, j);
					#f_new = new_new_solve_all(damper_x, wt_new);
					f_new = pm_f(damper_x, wt_new, wt_limits, r, m, p);
			
					if (f_new < f_prev)
						wt_prev = wt_new;	# success
					endif
						
				endif
				
			end
			
		end
	
		
		printf("Iteration #%d w(t):\n", iter);
		disp(wt_prev);
		disp(' ');
		printf("Iteration #%d h(t):\n", iter);
		disp(h);
		disp(' ');
	
	
		#f_new = new_new_solve_all(damper_x, wt_prev);
		f_new = pm_f(damper_x, wt_prev, wt_limits, r, m, p);
	
		do_next = false;
		if (f_new < f_orig)
			#wt_prev = wt_prev + lambda * (wt_prev - wt_orig);
			#f_orig = new_new_solve_all(damper_x, wt_prev);
			f_orig = pm_f(damper_x, wt_prev, wt_limits, r, m, p);
			do_next = true;
		else
			if (h > vareps)
				printf("%f -> ", h);
				h = h / hj_alpha;
				printf("%f\n\n", h);
				do_next = true;
			endif
		endif
				
		printf("Iteration #%d error: %f\n", iter, f_orig);
		disp(' ');
		if (do_next == false)
			break;
		endif
	end
	wt = wt_prev;
endfunction

