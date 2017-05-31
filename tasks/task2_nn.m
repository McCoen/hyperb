source("damping/newton.m");
source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

global boundary = @(x) 0.25 * sin(pi * x);

global task = 1;

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

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan];

global enable_interpolation = false;
global full_diff = true;

global n = 160;
global k = 64;
global wt_start = zeros(k + 1, num_of_dampers);
%  t = newton(0.1, 0.2, damper_x, wt_start);
t = 0.1;
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
global wt = [-12.9145
  -52.2668
  -52.7763
  -36.4827
  -59.1695
  -54.0746
  -65.0549
  -49.8526
  -63.9771
  -46.9206
  -69.9820
  -83.6936
  -36.8928
  -15.2750
  -39.1235
  -37.7765
  -46.2728
  -41.3093
  -25.3620
   15.5087
    1.6274
  -12.9776
   10.2115
    8.2446
   -3.8861
   12.9799
  -18.1192
  -28.2704
   -6.1076
   -5.9343
    1.6792
   39.0324
    5.2251
    1.6955
   22.9639
   26.1787
   14.5918
   25.1271
   30.7878
   17.2970
    6.1558
    1.0737
   14.1264
   13.7127
   25.3607
   33.0855
   20.9351
   30.2711
   36.1126
   36.1904
   60.2524
   49.1651
   62.1391
   59.2990
   61.8848
   61.6598
   88.8321
   79.1457
   39.9372
   53.4278
   80.0184
   65.3902
   80.7110
   41.4963
    3.3680];
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
plot_wt(wt, t, 1);
