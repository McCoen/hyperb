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

global a = 1;
global l = 1;
global t_total = 0.08;

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

%plot_start_h(1);
%exit;

damper_x = [0.5];
wt_bounds = [nan, nan; nan, nan];

wt = [-9.9376
  -215.3047
  -492.7332
  -267.9763
   -70.5620
  -309.4895
  -323.3094
   112.6743
   368.6876
   545.0719
   506.9271
    78.7342
  -168.2066
   190.2821
   322.0639
  -176.4228
  -586.7108
  -284.7539
   118.6413
   -70.0530
  -496.7566
  -343.8716
   105.9184
   367.2258
   514.0291
   429.4834
   121.2265
   213.3421
   451.4942
    82.4290
  -373.7194
  -189.0911
   -18.7667];

print_dampers_info(damper_x, wt_bounds, wt);
disp('E(T) = ');
disp(solve_all(damper_x, wt));
disp('');

print_u_grid_values();

#plot_u_grid(1);
#plot_f(1, damper_x, wt);
#plot_wt(1, wt, 1);
