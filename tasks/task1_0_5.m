source('damping/hyperb.m');

global h0 = @(x) 0.1 * sin(2 * pi * x);
global h1 = @(x) 0.0;

global task = 1;

global n = 20;
global k = 250;
global vareps = power(10, -16);

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan];

global st = zeros(k + 1, num_of_dampers);

a = 1;
l = 1;
t = 0.5;

hx = l / n;
ht = t / k;

print_task_init(a, l, t, hx, ht);

u = osc_u(hx);
v = osc_v(a, l, hx);

print_to_eps = true;

wt = zeros(k + 1, num_of_dampers);
err = gpuEnergyInt(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);
u = gpuOscDeadening(damper_x, wt, st, num_of_dampers, n, k, u, v, a, l, t);

disp('E(T) = ');
disp(err);
disp('');

%plot_grid_black_and_white(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
%plot_wt_black_and_white(wt, wt_bounds, t, '-S480, 320', print_to_eps);
plot_grid(err, a, l, t, hx, ht, u, 100, '-S480, 320', print_to_eps);
%plot_wt(wt, wt_bounds, t, '-S480, 320', print_to_eps);