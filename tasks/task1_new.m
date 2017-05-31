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

global vareps = power(10, -6);

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan];

global enable_interpolation = false;
global full_diff = true;

global n = 160;
global k = 64;
global wt_start = zeros(k + 1, num_of_dampers);

t = 0.08;
hx = l / n;
ht = t / k;

global eta = power(10, -3);
global xi = power(10, -4);

print_task_init(t, hx, ht);

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

global wt = wt_start;
for i = 1 : length(wt(:))
	wt(i) = 1;%(rand(1) - rand(1)) * 10;
endfor
print_dampers_info();

wt = [-1.0992
  -158.0818
  -226.3690
  -197.7487
  -159.8911
  -138.0688
   -85.0434
  -109.6649
  -108.6313
   -77.4381
   -57.6477
    -6.8345
   -52.6084
   -27.7361
    34.2842
   -51.0517
    12.2149
    46.3783
   -67.7164
   -64.8048
   -13.2725
     4.1706
    24.5389
    34.3782
   111.9169
   224.5429
   148.9522
    45.1938
    32.1645
    15.2084
    28.8081
    28.0255
    -4.7335
     8.2408
   -15.6391
   -53.8079
   -48.5960
    -8.6916
  -112.6507
  -167.4873
  -158.3013
  -103.8289
   -11.2431
    75.1174
    57.0904
    88.4446
    32.0207
    21.2652
    54.4944
    61.2855
    13.3682
   108.3076
    34.2005
    16.6560
    69.9099
    89.8352
    65.0383
    95.3842
    64.6563
   164.8831
    97.8623
   206.8559
   280.5244
   150.8092
   -18.0948];
%wt = marquardt(@solve_all, damper_x, wt, 1500, vareps, power(10, 4), hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta, false);

print_dampers_info();
disp('E(T) = ');
alpha = osc_alpha(hx, ht);
beta = osc_beta(hx);

u = oscDeadening(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
err = energyInt(damper_x, wt, n, k, hx, ht, u, v, c, c_inverted, sweep_side, cTilde, beta);
disp(err);
disp('');

print_u_grid_values(ht, u);

plot_u_grid(err, t, hx, ht, u);
#plot_f(1, damper_x, wt);
%plot_wt(wt, 1);
