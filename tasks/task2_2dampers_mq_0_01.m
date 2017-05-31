source("damping/osc_deadening.m");
source("damping/grid_integration.m");
source("damping/simplified_tridiag_matrix_sweep.m");
source("minimization/hooke_jeeves.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/simulated_annealing.m");
source("minimization/mq_hessian.m");

function boundary = boundary(x)
	boundary = 0.1 * sin(2 * pi * x);
endfunction

global task = 2;

global a = 1;
global l = 1;
global t_total = 0.01;

global vareps = power(10, -10);

global n = 40;
global k = 16;
global hx = l / n;
global ht = t_total / k;

global num_of_dampers = 2;
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
disp('0.1 * sin(2 * pi * x)');
disp('');

#plot_start_h(1);
#pause;

damper_x = [0.25, 0.75];
wt_bounds = [nan, nan; nan, nan];

#wt = marquardt(@solve_all, damper_x, wt_start, Inf, vareps);
wt = [51.5127,   -50.2213;
    80.0610,   -80.1493;
    -8.1172,     4.3225;
  -106.2683,   118.0353;
    79.1302,  -103.3611;
    47.1881,    -7.9189;
   159.2607,  -211.7284;
   -34.2489,    91.4206;
    48.7500,   -94.9910;
   126.4735,  -112.3422;
   112.7745,   -71.8901;
  -170.3234,    54.8875;
   249.3582,   -47.8189;
  -336.4900,    47.8448;
   390.9203,   -23.9218;
   -60.4179,  -370.6354;
  -137.2152,   618.6453];

print_dampers_info(damper_x, wt_bounds, wt);
disp('E(T) = ');
disp(solve_all(damper_x, wt));
#solve_all(damper_x, wt);
printf("\n");

print_u_grid_values();

#plot_u_grid();
plot_wt(wt, 1);
