source("damping/newton.m");
source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

global boundary = @(x) 0.1 * sin(2 * pi * x);

global task = 2;

global a = 1;
global l = 1;


%  1.4138e-06 0.2
%  1.4269e-06 0.19
%  7.5360e-06 0.15
%  4.9173e-04 0.12
%  0.0072092 0.1

global t_a = 0.1;
global t_b = 0.2;
%  global t_total = 0.1;

global vareps = power(10, -6);

global num_of_dampers = 2;
global damper_x = [0.25, 0.75];
global wt_bounds = [nan, nan; nan, nan];

global enable_interpolation = false;
global full_diff = true;

global n = 80;
global k = 32;
global wt_start = zeros(k + 1, num_of_dampers);
%  t = newton(0.1, 0.2, damper_x, wt_start);
t = 0.08;
hx = l / n;
ht = t / k;

global eta = power(10, -3);
global xi = power(10, -4);

print_task_init(t, hx, ht);

%plot_start_h(1);
%exit;

%wt = marquardt_continue(@solve_all, damper_x, 'marquardt_last', 5, vareps);
alpha = osc_alpha(hx, ht);
beta = osc_beta(hx);

b = osc_b();
c = osc_c(alpha, b);
c_inverted = inv(c);
sweep_side = eye(2);

cTilde = osc_cTilde(alpha, b);

u = osc_y(hx);
v = osc_sec_y();

%global wt = wt_start;
global wt = [0.50603,   -0.58570;
    2.57409,   -2.56080;
    5.11786,   -5.11790;
    7.27865,   -7.17016;
    9.26214,   -9.47961;
   11.35606,  -11.28599;
   13.67940,  -13.78188;
   16.14066,  -16.04339;
   18.01372,  -18.05163;
   19.38521,  -19.35649;
   20.83805,  -20.80382;
   21.99128,  -21.80015;
   22.33345,  -22.35595;
   23.59571,  -23.66166;
   24.48330,  -24.47579;
   24.91359,  -24.91088;
   25.13850,  -25.17370;
   24.74374,  -24.78332;
   24.48987,  -24.50273;
   23.28882,  -23.35301;
   22.28061,  -22.22070;
   21.99554,  -22.00495;
   20.52959,  -20.59569;
   19.16644,  -19.20394;
   17.93116,  -17.88845;
   15.69630,  -15.57664;
   13.29452,  -13.25135;
   11.06200,  -11.06498;
    8.98393,   -8.96792;
    7.05543,   -6.96038;
    4.68827,   -4.69974;
    2.01763,   -2.17639;
    0.18576,   -0.21387];
%wt = marquardt(@solve_all, damper_x, wt_start, 100, vareps, power(10, 4), hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta, false);

%  disp(ei);
%  ei = power(10, -4) - solve_all(wt, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
%  wt = marquardt(@solve_all, damper_x, wt_start, 100, vareps, power(10, 4), false);

print_dampers_info();
disp('E(T) = ');
alpha = osc_alpha(hx, ht);
beta = osc_beta(hx);

%  	wt = marquardt(@solve_all, damper_x, wt_start, 100, vareps, power(10, 4), hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta, true);
	%err = power(10, -4) - solve_all(wt, hx, ht, y, y_sec, c, c_inverted, sweep_side, cTilde, beta);
	
u = oscDeadening(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
err = energyInt(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
%  extended_print_grid_to_file(u, n, k, hx, ht, "absOut");
disp(err);
%printf("%.16f\n", err);
disp('');

%  print_u_grid_values(ht, u);

%plot_u_grid(err, t, hx, ht, u);
#plot_f(1, damper_x, wt);
%plot_wt(wt, t, 2);
