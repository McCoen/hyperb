1;

function mq_hessian(shared, grad, damper_x, wt, der_h, nth, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)
	global num_of_cores;

	for i = nth : num_of_cores : length(wt(:))

		d0 = grad(i);

		for j = i : length(wt(:))

			wt1 = wt;
	
			wt1(j) = wt1(j) + der_h;
	
			d1 = derivative_at(damper_x, wt1, i, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	
			sd = (d1 - d0) / der_h;
		
			fprintf(shared, "%.256e\n", sd);
		endfor

	endfor
	fclose(shared);
endfunction
