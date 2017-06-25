source("damping/newton.m");
source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

global boundary = @(x) 0.1 * sin(2 * pi * x);

global task = 2;

global n = 80;
global k = 32;
global vareps = power(10, -6);

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan];
global wt_start = zeros(k + 1, num_of_dampers);

global st = zeros(k + 1, num_of_dampers);
st_max_variation = 0.05;
st_min_variation = -0.05;
for i = 1 : k + 1
	if (mod(i, 2) == 1)
		st(i) = st_max_variation;
	else
		st(i) = st_min_variation;
	endif
endfor
disp(st);

global enable_interpolation = false;
global full_diff = true;

global a = 1;
global l = 1;
t = 0.4;

hx = l / n;
ht = t / k;

global eta = power(10, -3);
global xi = power(10, -4);

print_task_init(t, hx, ht);

u = osc_y(hx);

wt = [-0.88636
    5.58253
  -25.90064
   30.89409
  -34.48905
   18.77461
    1.74026
  -16.52242
   31.33720
  -36.66408
   27.02482
  -20.27076
    3.20397
    3.07243
  -14.87118
   30.92046
  -34.88391
   33.77421
  -22.07300
   12.19013
   -4.74746
  -13.93650
   22.15924
  -31.00917
   35.28518
  -22.44674
    8.19637
   14.55969
  -33.65319
   41.63935
  -41.69925
   11.75288
   -0.76874];
wt = gpuMarquardt(damper_x, wt, st, num_of_dampers, n, k, u, 1, 1, t, 0, vareps);
disp(wt);
disp('');

u = oscDeadening(damper_x, wt, st, n, k, u, 1, 1, t, 0, vareps);
extended_print_grid_to_file(u, t, n, k, hx, ht, "test.m");
disp(u);
%disp('');
exit;
