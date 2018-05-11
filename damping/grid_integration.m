1;

global spl_x spl_t;

function der = u_x(nx)

	global enable_interpolation n k y spl_x hx eta;

	if (enable_interpolation)
	
		if (nx == 1)
			f1 = spl_x(2);
			f0 = spl_x(1);
			der = (f1 - f0) / eta;
		elseif (nx == length(spl_x))
			f1 = spl_x(length(spl_x));
			f0 = spl_x(length(spl_x) - 1);
			der = (f1 - f0) / eta;
		else
			f2 = spl_x(nx + 1);
			f0 = spl_x(nx - 1);
			der = (f2 - f0) / (2 * eta);
		endif

	else

		if (nx == 1)
			f1 = y(k + 1, 2);
			f0 = y(k + 1, 1);
			der = (f1 - f0) / hx;
		elseif (nx == n + 1)
			f1 = y(k + 1, n + 1);
			f0 = y(k + 1, n);
			der = (f1 - f0) / hx;
		else
			f2 = y(k + 1, nx + 1);
			f0 = y(k + 1, nx - 1);
			der = (f2 - f0) / (2 * hx);
		endif

	endif
endfunction

function der = u_t(nx)

	global enable_interpolation k y spl_t ht xi;

	if (enable_interpolation)
		f = spl_t(:, nx);
		%der = (f(1) - f(2)) / xi;
		der = (f(2) - f(1)) / xi;
	else
		f1 = y(k, nx);
		f2 = y(k + 1, nx);
		der = (f2 - f1) / ht;
	endif
endfunction

function ei = energy_int()
	global hx l t_total;
	global a y xr yr;

	global enable_interpolation full_diff;
	global eta xi spl_x spl_t;

	if (enable_interpolation)
		spl_x = interp2(xr, yr, y, 0.0 : eta : l, t_total, "spline");
		spl_t = interp2(xr, yr, y, 0.0 : eta : l, [t_total - xi, t_total], "spline");

		int_x = 0.0 : eta : l;
	else
		int_x = 0.0 : hx : l;
	endif

	int_y = zeros(length(int_x));
	for nx = 1 : length(int_x)
		if (full_diff)
			int_y(nx) = power(u_t(nx), 2) + power(a, 2) * power(u_x(nx), 2);
		elseif (enable_interpolation)
			int_y(nx) = power(u_t(nx), 2) + power(a, 2) * power(spl_x(nx), 2);
		else
			int_y(nx) = power(u_t(nx), 2) + power(a, 2) * power(y(k + 1, nx), 2);
		endif
	endfor

	ei = trapz(int_x, int_y);
	ei = ei(1);
	
endfunction
