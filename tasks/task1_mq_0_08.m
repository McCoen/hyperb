source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

function boundary = boundary(x)
	boundary = 0.25 * sin(pi * x);
endfunction

global task = 1;

global a = 1;
global l = 1;
global t_total = 0.1;

global vareps = power(10, -10);

global n = 256;
global k = 128;

global num_of_dampers = 1;
global damper_x = [0.5];
global wt_bounds = [nan, nan];

global enable_interpolation = true;
global full_diff = true;

global hx = l / n;
global ht = t_total / k;
global wt = zeros(k + 1, num_of_dampers);

print_task_init();

%plot_start_h(1);
%exit;

wt = [0.25498
   2.89621
   5.15282
   3.45552
   1.54012
   2.74363
   4.24900
   4.57871
   3.21303
   4.87810
   3.37174
   1.85742
   2.83764
   3.14140
   2.83667
   4.74808
   3.44382
   3.62951
   3.83881
   3.85795
   3.02247
   4.71096
   2.80607
   4.15087
   2.60352
   2.17336
   3.83305
   4.96579
   3.35119
   4.82363
   4.57257
   3.76980
   3.90398
   2.44604
   4.08864
   3.62915
   4.23435
   3.22851
   4.93331
   5.89963
   4.67414
   5.16609
   4.57354
   6.27775
   4.76022
   3.31444
   5.11509
   4.37786
   3.32939
   4.70204
   3.61171
   3.25273
   5.73169
   4.18868
   6.12624
   6.10644
   6.14166
   4.71839
   4.64025
   4.92558
   4.44506
   7.39314
   3.94290
   5.52834
   5.40554
   7.44383
   5.13947
   6.57514
   5.42421
   4.61258
   4.30354
   6.24465
   5.56491
   6.21526
   6.95204
   5.33827
   2.63853
   4.35157
   3.81822
   3.80783
   6.04753
   5.28150
   6.05229
   7.44374
   5.20134
   4.40151
   6.04173
   5.70341
   4.30195
   5.31083
   4.33976
   5.96272
   6.43457
   5.55676
   6.56693
   5.21789
   4.91579
   5.29585
   6.74259
   5.48891
   4.78964
   6.28059
   4.22309
   4.67414
   2.36930
   5.57655
   6.21237
   5.85216
   5.88765
   6.69243
   5.45236
   5.75541
   5.75312
   4.07742
   5.33569
   5.72460
   5.74938
   5.11395
   5.57444
   5.55471
   6.33426
   6.07544
   4.94003
   6.50115
   5.07465
   6.88992
   6.50396
   3.45732
   1.67707];
%  wt = marquardt(@solve_all, damper_x, wt, Inf, vareps, power(10, 4));

print_dampers_info();
disp('E(T) = ');
err = solve_all(damper_x, wt);
disp(err);
disp('');

print_u_grid_values();

plot_u_grid(err);
#plot_f(1, damper_x, wt);
%plot_wt(wt, 1);
