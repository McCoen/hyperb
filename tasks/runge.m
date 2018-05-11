source('damping/hyperb.m');

global h0 = @(x) 0.25 * sin(pi * x);
global h1 = @(x) 0.0;

global task = 1;

global vareps = power(10, -6);

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan];

global n = 20;
global k = 20;

global st = zeros(k + 1, num_of_dampers);

a = 1;
l = 1;
t = 0.1;

hx = l / n;
ht = t / k;

print_task_init(a, l, t, hx, ht);

u = osc_u(hx);
v = osc_v(a, l, hx);

wt = zeros(k + 1, num_of_dampers);
mq_max_iterations = inf;
wt = gpuMarquardtMinimization(damper_x, wt_bounds, wt, st, num_of_dampers, n, k, u, v, a, l, t, mq_max_iterations, vareps);
err_1 = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);






n = 40;
k = 40;

global st = zeros(k + 1, num_of_dampers);

a = 1;
l = 1;
t = 0.1;

hx = l / n;
ht = t / k;

print_task_init(a, l, t, hx, ht);

u = osc_u(hx);
v = osc_v(a, l, hx);

wt = zeros(k + 1, num_of_dampers);
mq_max_iterations = inf;
wt = gpuMarquardtMinimization(damper_x, wt_bounds, wt, st, num_of_dampers, n, k, u, v, a, l, t, mq_max_iterations, vareps);
err_2 = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);






n = 80;
k = 80;

global st = zeros(k + 1, num_of_dampers);

a = 1;
l = 1;
t = 0.1;

hx = l / n;
ht = t / k;

print_task_init(a, l, t, hx, ht);

u = osc_u(hx);
v = osc_v(a, l, hx);

wt = zeros(k + 1, num_of_dampers);
mq_max_iterations = inf;
wt = gpuMarquardtMinimization(damper_x, wt_bounds, wt, st, num_of_dampers, n, k, u, v, a, l, t, mq_max_iterations, vareps);
err_3 = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);

disp('E(T) 20 x 20 = ');
disp(err_1);
disp('');

disp('E(T) 40 x 40 = ');
disp(err_2);
disp('');

disp('E(T) 80 x 80 = ');
disp(err_3);
disp('');


runge = abs((err_1 - err_2) / (err_2 - err_3));

disp('Runge function = ');
disp(runge);
disp('');

%{
E(T) 20 x 20 =
  1.3293e-008

E(T) 40 x 40 =
  2.7199e-011

E(T) 80 x 80 =
  2.1102e-009

Runge function =
 6.3686
%}