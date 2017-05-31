source("damping/new_osc_deadening.m");
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

global a = 1;
global l = 1;
global t_total = 0.12;

global vareps = power(10, -10);

global n = 40;
global k = 16;
global hx = l / n;
global ht = t_total / k;

global num_of_dampers = 1;
global wt_start = zeros(k + 1, num_of_dampers);
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

#{
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
#}

#plot_start_h(1);
#pause;

damper_x = [0.5];
wt_bounds = [nan, nan; nan, nan];

wt = marquardt(@solve_all, damper_x, wt_start, -1, vareps);

#print_dampers_info(damper_x, wt_limits, wt);
#disp('E(T) = ');
disp(solve_all(damper_x, wt));
#solve_all(damper_x, wt);
#printf("\n");

print_u_grid_values();

#plot_u_grid("u(x, t)_0_1_16x40.png");
#plot_wt(1, wt, 1);
