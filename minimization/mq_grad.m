1;

function der = derivative_at(damper_x, wt, nth, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)
	global mq_fun;

	der_h = power(10, -4);

	wt1 = wt;
	
	wt1(nth) = wt1(nth) + der_h;
	
	f0 = mq_fun(wt, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	f1 = mq_fun(wt1, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	
	der = (f1 - f0) / der_h;

endfunction

function mq_grad(shared, damper_x, wt, der_h, nth, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta)
	global num_of_cores;

	for i = nth : num_of_cores : length(wt(:))
		grad = derivative_at(damper_x, wt, i, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
		fprintf(shared, "%.256e\n", grad);
	endfor
	fclose(shared);
endfunction
