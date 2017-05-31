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
global t_total = 0.1;

global vareps = power(10, -10);

global n = 64;
global k = 32;
global hx = l / n;
global ht = t_total / k;

global enable_interpolation = true;
global full_diff = true;

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

disp('Параметры сетки:');
printf("n = %d\n", n);
printf("k = %d\n", k);
disp('');

printf("hx = %f\n", hx);
printf("ht = %f\n", ht);
disp('');

if (enable_interpolation)
	global eta = power(10, -3);
	global xi = power(10, -4);

	disp('Шаги дифференцирования/интегрирования:');
	printf("η = %e\n", eta);
	printf("ξ = %e\n", xi);
	disp('');
endif

disp('Начальное возмущение =');
disp('0.25 * sin(pi * x)');
disp('');

%plot_start_h(1);
%exit;

damper_x = [0.5];
wt_bounds = [nan, nan; nan, nan];

wt = [10.5296
   -52.4444
   -76.9410
   -78.1322
   -30.9095
   -72.5723
   -41.8684
   -44.6156
   -28.0121
   -43.5542
    -5.8049
   -27.2549
    29.4326
    33.9371
    23.9675
   -28.9591
    33.9777
    27.3350
    17.2408
   -21.7461
    17.3276
    -8.3731
    29.1172
    13.6924
    39.8325
    47.1207
   110.2430
    55.0596
    76.8519
    81.5546
    93.0259
    13.5597
    15.2144];

print_dampers_info(damper_x, wt_bounds, wt);
disp('E(T) = ');
err = solve_all(damper_x, wt);
disp(err);
disp('');

%print_u_grid_values();

%plot_u_grid(err);
#plot_f(1, damper_x, wt);
plot_wt(wt, 1);
