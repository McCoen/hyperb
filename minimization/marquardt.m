1;

global mq_fun;

global num_of_cores = 2;
global all_pids;

function recursive_grad_proc(damper_x, wt, der_h, nth, max_nth, current_pid, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)

	global all_pids;

	shared = fopen(strcat('marquardt_shared_', int2str(nth)), 'w');

	[pid, msg] = fork();
	all_pids(current_pid) = pid;

	if pid
		if (nth < max_nth)
			recursive_grad_proc(damper_x, wt, der_h, nth + 1, max_nth, current_pid + 1, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)
		endif
	else
		if (nth <= max_nth)
			mq_grad(shared, damper_x, wt, der_h, nth, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
		endif
		exit;
	endif
endfunction

function grad = new_gradient_at(damper_x, wt, total_iter, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)

	global num_of_cores all_pids;
	
	grad = zeros(length(wt(:)), 1);

	n = length(wt(:));
	
	der_h = power(10, -4);

	all_pids = zeros(num_of_cores, 1);
	current_id = 1;

	recursive_grad_proc(damper_x, wt, der_h, 1, num_of_cores, 1, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);


	for i = 1 : num_of_cores

		waitpid(all_pids(i));

	endfor

	for nth = 1 : num_of_cores

		shared = fopen(strcat('marquardt_shared_', int2str(nth)), 'r');
		for i = nth : num_of_cores : length(wt(:))
			grad(i) = str2double(fgetl(shared));
		endfor
		fclose(shared);
	endfor
endfunction

function recursive_hessian_proc(grad, damper_x, wt, der_h, nth, max_nth, current_pid, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)
	global all_pids;

	shared = fopen(strcat('marquardt_shared_', int2str(nth)), 'w');

	[pid, msg] = fork();
	all_pids(current_pid) = pid;

	if pid
		if (nth < max_nth)
			recursive_hessian_proc(grad, damper_x, wt, der_h, nth + 1, max_nth, current_pid + 1, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)
		endif
	else
		if (nth <= max_nth)
			mq_hessian(shared, grad, damper_x, wt, der_h, nth, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
		endif
		exit;
	endif
endfunction


function h = mproc_hessian_at(grad, damper_x, wt, total_iter, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)

	global num_of_cores all_pids;

	n = length(wt(:));
	h = zeros(n, n);
	
	iter = 0;
	total = power(n, 2);
	
	der_h = power(10, -4);

	all_pids = zeros(num_of_cores, 1);
	current_id = 1;

	recursive_hessian_proc(grad, damper_x, wt, der_h, 1, num_of_cores, 1, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);

	for i = 1 : num_of_cores

		waitpid(all_pids(i));

	endfor


	for nth = 1 : num_of_cores

		shared = fopen(strcat('marquardt_shared_', int2str(nth)), 'r');
		for i = nth : num_of_cores : length(wt(:))


			for j = i : length(wt(:))

				sd = str2double(fgetl(shared));
				h(i, j) = sd;
				h(j, i) = sd;
			endfor

		endfor
		fclose(shared);
	endfor
endfunction

function n = euclidean_norm(grad)
	n = 0;
	for i = 1 : length(grad)
		n = n + power(grad(i), 2);
	end
	n = sqrt(n);
endfunction

function wt = marquardt_continue(fun, damper_x, filename, m, vareps)
	global k num_of_dampers;

	input_file = fopen(filename, 'r');

	mu_temp = str2double(fgetl(input_file));
	
	wt_temp = zeros(k + 1, num_of_dampers);
	for j = 1 : num_of_dampers
		for i = 1 : k + 1
			wt_temp(i, j) = str2double(fgetl(input_file));
		endfor
	endfor
	fclose(input_file);

	wt = marquardt(fun, damper_x, wt_temp, m, vareps, mu_temp);
endfunction

function wt = marquardt(fun, damper_x, wt_start, m, vareps, start_mu, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta, silent)
	global num_of_dampers mq_fun;

	k = length(wt_start(:, 1)) - 1;

	printf("\nUsing Marquardt minimization\n\n");

	mq_fun = fun;

	iter = 0;
	muNotInf = true;
	
	f_prev = mq_fun(wt_start, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);

	if (silent == false)
		printf("Iteration %d/%d error:\n", iter, m);
		disp(f_prev);
		printf("\n");
	endif
	
	wt = wt_start;
	
 	grad = new_gradient_at(damper_x, wt, iter, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
# 	global n k hx ht y y_sec c c_inverted sweep_side cTilde beta;
# 	grad = gpuGradientAt(damper_x, wt, n, k, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	if (silent == false)
		printf("Iteration %d/%d gradient:\n", iter, m);
		disp(grad);
		printf("\n");
	endif

	grad_norm = euclidean_norm(grad);
	if (silent == false)
		printf("Iteration %d/%d gradient norm:\n", iter, m);
		disp(grad_norm);
		printf("\n");
	endif

	while (grad_norm > vareps && iter < m && muNotInf == true)
	
	
		iter = iter + 1;
		
  		h = mproc_hessian_at(grad, damper_x, wt, iter, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
# 		h = gpuHessianAt(damper_x, wt, n, k, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta, grad);
		
		if (silent == false)
			printf("Iteration %d/%d Hessian matrix:\n", iter, m);
			disp(h);
			printf("\n");
		endif
		
		if (iter == 1)
			mu = start_mu;
		endif

		if (silent == false)
			printf("Iteration %d/%d μ:\n", iter, m);
			disp(mu);
			printf("\n");
		endif
		
		do
	
			muEye = eye(length(wt(:))) * mu;
	
			hMuEye = h + muEye;
	
			hMuEyeInv = inv(hMuEye);

			d = -1 * hMuEyeInv * grad;
			if (silent == false)
				printf("Iteration %d/%d d:\n", iter, m);
				disp(d);
				printf("\n");
			endif

			d_i = 1;
			wt_d = zeros(k + 1, num_of_dampers);
			for wt_i = 1 : num_of_dampers
				for wt_j = 1 : k + 1
					wt_d(wt_j, wt_i) = d(d_i);
					d_i = d_i + 1;
				end
			end
			
			wt_new = wt + wt_d;
			
			f_new = mq_fun(wt_new, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	
			if (f_new < f_prev)
				if (silent == false)
					printf("μ = %f / 2 -> %f\n\n", mu, mu / 2);
				endif
				mu = mu / 2;
			else
				if (silent == false)
					printf("μ = %f * 2 -> %f\n\n", mu, mu * 2);
				endif
				mu = mu * 2;
			endif
			
			if (mu == inf)
				muNotInf = false;
				break;
			endif
			
		
		until (f_new < f_prev)
		
		f_prev = f_new;
		
		if (silent == false)
			printf("Iteration %d/%d w(t):\n", iter, m);
			disp(wt_new);
			printf("\n");
		endif

		output_file = fopen('marquardt_last', 'w');
		fprintf(output_file, "%.256e\n", mu);
		for i = 1 : length(wt(:))
			fprintf(output_file, "%.256e\n", wt_new(i));
		endfor
		fclose(output_file);
	
		if (silent == false)
			printf("Iteration %d/%d error:\n", iter, m);
			disp(f_new);
			printf("\n");
		endif
		
		wt = wt_new;
				
		grad = new_gradient_at(damper_x, wt, iter, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
# 		grad = gpuGradientAt(damper_x, wt, n, k, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
		if (silent == false)
			printf("Iteration %d/%d gradient:\n", iter, m);
			disp(grad);
			printf("\n");
		endif
		
		grad_norm = euclidean_norm(grad);
		if (silent == false)
			printf("Iteration %d/%d gradient norm:\n", iter, m);
			disp(grad_norm);
			printf("\n");
		endif
	endwhile
endfunction
