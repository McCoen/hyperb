source("damping/osc_deadening.m");
source("damping/grid_integration.m");
source("damping/simplified_tridiag_matrix_sweep.m");
source("minimization/hooke_jeeves.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/simulated_annealing.m");
source("minimization/mq_hessian.m");

function boundary = boundary(x)
	boundary = 0.25 * sin(pi * x);
endfunction

global task = 1;

global a = 1;
global l = 1;
global t_total = 0.2;

global vareps = power(10, -10);

global n = 80;
global k = 32;
global hx = l / n;
global ht = t_total / k;

global num_of_dampers = 1;
global wt = zeros(k + 1, num_of_dampers);
global damper_x = zeros(1, num_of_dampers);
global wt_bounds = zeros(num_of_dampers, 2);

global alpha = osc_alpha();
global beta = osc_beta();

global b = osc_b();
global c = osc_c();
global c_inverted = inv(c);
global sweep_side = eye(2);
global cTilde = osc_cTilde();

global y = osc_y();
global y_sec = osc_sec_y();

printf("a = %f\n", a);
printf("l = %f\n", l);
printf("T = %f\n", t_total);

printf("\n");
disp('ε =');
disp(vareps);
printf("\n");

printf("n = %d\n", n);
printf("k = %d\n", k);
printf("hx = %f\n", hx);
printf("ht = %f\n", ht);
printf("\n");

disp('Начальное возмущение =');
disp('0.25 * sin(pi * x)');
disp('');

#plot_start_h(1);
#pause;

damper_x = [0.5];
wt_bounds = [nan, nan];

function pwt = new_probable_wt(nth, probable_t)
	global ht;

	t = ht * (nth - 1);
	pwt = probable_t(34) + probable_t(nth) * cos(probable_t(35) * t + probable_t(36));

	#2.1930
endfunction

function pwt = probable_wt(nth, x)
	global ht;

	t = ht * (nth - 1);
	pwt = x(1) + x(2) * cos(x(3) * t + x(4));

	#2.1930
endfunction

x_start = [0; 1; 1; 0];

start_probable_t = zeros(k + 1 + 3, 1);

function res = new_probable_mq(damper_x, probable_t)
	global wt;

	wt = [new_probable_wt(1, probable_t);
		new_probable_wt(2, probable_t);
		new_probable_wt(3, probable_t);
		new_probable_wt(4, probable_t);
		new_probable_wt(5, probable_t);
		new_probable_wt(6, probable_t);
		new_probable_wt(7, probable_t);
		new_probable_wt(8, probable_t);
		new_probable_wt(9, probable_t);
		new_probable_wt(10, probable_t);
		new_probable_wt(11, probable_t);
		new_probable_wt(12, probable_t);
		new_probable_wt(13, probable_t);
		new_probable_wt(14, probable_t);
		new_probable_wt(15, probable_t);
		new_probable_wt(16, probable_t);
		new_probable_wt(17, probable_t);
		new_probable_wt(18, probable_t);
		new_probable_wt(19, probable_t);
		new_probable_wt(20, probable_t);
		new_probable_wt(21, probable_t);
		new_probable_wt(22, probable_t);
		new_probable_wt(23, probable_t);
		new_probable_wt(24, probable_t);
		new_probable_wt(25, probable_t);
		new_probable_wt(26, probable_t);
		new_probable_wt(27, probable_t);
		new_probable_wt(28, probable_t);
		new_probable_wt(29, probable_t);
		new_probable_wt(30, probable_t);
		new_probable_wt(31, probable_t);
		new_probable_wt(32, probable_t);
		new_probable_wt(33, probable_t)];

	res = solve_all(damper_x, wt);
endfunction

function res = probable_mq(damper_x, x)
	global wt;

	wt = [probable_wt(1, x);
		probable_wt(2, x);
		probable_wt(3, x);
		probable_wt(4, x);
		probable_wt(5, x);
		probable_wt(6, x);
		probable_wt(7, x);
		probable_wt(8, x);
		probable_wt(9, x);
		probable_wt(10, x);
		probable_wt(11, x);
		probable_wt(12, x);
		probable_wt(13, x);
		probable_wt(14, x);
		probable_wt(15, x);
		probable_wt(16, x);
		probable_wt(17, x);
		probable_wt(18, x);
		probable_wt(19, x);
		probable_wt(20, x);
		probable_wt(21, x);
		probable_wt(22, x);
		probable_wt(23, x);
		probable_wt(24, x);
		probable_wt(25, x);
		probable_wt(26, x);
		probable_wt(27, x);
		probable_wt(28, x);
		probable_wt(29, x);
		probable_wt(30, x);
		probable_wt(31, x);
		probable_wt(32, x);
		probable_wt(33, x)];

	res = solve_all(damper_x, wt);
endfunction

#x = marquardt(@probable_mq, damper_x, x_start, Inf, vareps);
#probable_mq(damper_x, x);

probable_t = [4.116957
   -1.024626
   -2.618143
    0.424756
   -0.365881
    2.021534
    1.412571
    4.120933
    3.801132
    5.842654
    9.396290
   11.031135
   13.102524
   13.377829
   14.789827
   12.478887
   14.671281
   15.421338
   14.379226
   15.891799
   16.175499
   16.852092
   19.542488
   21.583319
   21.312196
   20.912361
   19.543991
   20.535578
   18.125755
   18.518578
   19.524215
   15.028230
    8.038014
   -6.256954
   -0.175755
    0.021807];
new_probable_mq(damper_x, probable_t);

#probable_t = marquardt(@probable_mq, damper_x, start_start_probable_t, Inf, vareps);
#new_probable_mq(damper_x, probable_t);
#x = [4.7659; 9.2766; 21.6003; -3.4112];

print_dampers_info(damper_x, wt_bounds, wt);
disp('E(T) = ');
disp(solve_all(damper_x, wt));
disp('');

#print_u_grid_values();

#plot_u_grid();
plot_wt(wt, 1);
#plot_f(1, damper_x, wt);
#plot_beta(1, probable_t, 1);
