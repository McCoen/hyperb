source("damping/hyperb.m");
source("damping/grid_integration.m");
source("minimization/pen_minimization.m");
source("minimization/marquardt.m");
source("minimization/mq_hessian.m");
source("minimization/mq_grad.m");

function boundary = boundary(x)
	boundary = 0.1 * sin(2 * pi * x);
endfunction

global task = 1;

global a = 1;
global l = 1;
global t_total = 0.05;

global vareps = power(10, -10);

# global n = 16;
# global k = 8;

# global n = 24;
# global k = 12;

# global n = 32;
# global k = 16;

# global n = 48;
# global k = 24;

# global n = 64;
# global k = 32;

# global n = 96;
# global k = 48;

global num_of_dampers = 2;
global damper_x = [0.25, 0.75];
global wt_bounds = [nan, nan; nan, nan];

global enable_interpolation = true;
global full_diff = true;

u = cell(6, 1);

# # # # # 

global n = 16;
global k = 8;
global hx = l / n;
global ht = t_total / k;
global wt = zeros(k + 1, num_of_dampers);

global alpha = osc_alpha();
global beta = osc_beta();

global b = osc_b();
global c = osc_c();
global c_inverted = inv(c);
global sweep_side = eye(2);
global cTilde = osc_cTilde();

global y = osc_y();
global y_sec = osc_sec_y();

global xr = 0.0 : hx : l;
global yr = 0.0 : ht : t_total;

global eta = power(10, -3);
global xi = power(10, -4);

wt = [-80.9865,    16.3071;
      48.0943,    16.5729;
     -84.8802,    20.2312;
      55.9733,     8.6554;
      -6.4473,   -58.1544;
      84.5535,   -19.9865;
      48.1091,  -112.6344;
      23.6729,    40.8135;
      28.0565,   -92.5050];

disp(solve_all(damper_x, wt));
u(1) = y;

# # # # # 

n = 24;
k = 12;
hx = l / n;
ht = t_total / k;
wt = zeros(k + 1, num_of_dampers);

alpha = osc_alpha();
beta = osc_beta();

b = osc_b();
c = osc_c();
c_inverted = inv(c);
sweep_side = eye(2);
cTilde = osc_cTilde();

y = osc_y();
y_sec = osc_sec_y();

xr = 0.0 : hx : l;
yr = 0.0 : ht : t_total;

eta = power(10, -3);
xi = power(10, -4);

wt = [4.9894,   10.0078;
     -9.7876,   -5.2612;
    -10.3076,   25.4037;
    -34.3311,   19.1987;
    -16.9233,   32.1528;
     -4.0684,  -11.2912;
     39.7348,  -24.2554;
     48.2518,  -63.8594;
     78.2263,  -62.4400;
     57.6424,  -73.6285;
     53.6904,  -37.5288;
      4.7914,  -21.1408;
     11.8025,    4.7528];
     
disp(solve_all(damper_x, wt));
u(2) = y;

# # # # # 

n = 32;
k = 16;
hx = l / n;
ht = t_total / k;
wt = zeros(k + 1, num_of_dampers);

alpha = osc_alpha();
beta = osc_beta();

b = osc_b();
c = osc_c();
c_inverted = inv(c);
sweep_side = eye(2);
cTilde = osc_cTilde();

y = osc_y();
y_sec = osc_sec_y();

xr = 0.0 : hx : l;
yr = 0.0 : ht : t_total;

eta = power(10, -3);
xi = power(10, -4);

wt = [22.3713,   26.2472;
   -26.1896,  -22.4706;
     8.9447,   39.7347;
   -42.0994,   -6.5842;
    18.7832,   29.9577;
   -17.0325,  -31.7670;
    16.3093,   32.5505;
   -39.6592,   -9.2744;
    38.1442,   10.8613;
    26.7705,  -75.8695;
    84.2943,  -35.0746;
    17.9851,  -67.3139;
    73.5648,  -24.1250;
    35.6122,  -85.1895;
    84.3581,  -34.6749;
    -5.9208,  -43.8955;
    30.5320,   19.4513];
     
disp(solve_all(damper_x, wt));
u(3) = y;

# # # # # 

n = 48;
k = 24;
hx = l / n;
ht = t_total / k;
wt = zeros(k + 1, num_of_dampers);

alpha = osc_alpha();
beta = osc_beta();

b = osc_b();
c = osc_c();
c_inverted = inv(c);
sweep_side = eye(2);
cTilde = osc_cTilde();

y = osc_y();
y_sec = osc_sec_y();

xr = 0.0 : hx : l;
yr = 0.0 : ht : t_total;

eta = power(10, -3);
xi = power(10, -4);

 wt = [-7.24348,   -6.25573;
   -12.14173,   23.48393;
   -35.65551,   21.97767;
    -8.98282,   24.31219;
   -17.18217,    8.19479;
    -3.94516,   15.87319;
    -7.50308,   -7.04300;
     8.24529,    1.01693;
     5.79421,  -16.32447;
    23.67943,  -12.38838;
    11.16437,  -20.18330;
    20.39886,  -11.64509;
    19.05630,  -27.49469;
    29.49038,  -22.20586;
    22.68698,  -28.61071;
    35.99748,  -29.19333;
    41.11310,  -45.83327;
    46.63258,  -45.37907;
    41.75608,  -47.05785;
    49.12162,  -46.89952;
    44.46899,  -41.26885;
    47.62403,  -46.22729;
    57.06820,  -56.69705;
    26.09588,  -30.16743;
     4.48979,   -0.60398];
     
disp(solve_all(damper_x, wt));
u(4) = y;

# # # # # 

n = 64;
k = 32;
hx = l / n;
ht = t_total / k;
wt = zeros(k + 1, num_of_dampers);

alpha = osc_alpha();
beta = osc_beta();

b = osc_b();
c = osc_c();
c_inverted = inv(c);
sweep_side = eye(2);
cTilde = osc_cTilde();

y = osc_y();
y_sec = osc_sec_y();

xr = 0.0 : hx : l;
yr = 0.0 : ht : t_total;

eta = power(10, -3);
xi = power(10, -4);

 wt = [-1.1205,    4.4304;
     -9.0172,    2.7669;
    -24.3998,   24.1206;
    -17.9565,   13.8788;
    -15.8244,   25.7481;
    -18.5611,   21.3443;
     -6.3400,   15.6485;
     -4.8881,    1.6409;
    -11.8306,   10.4163;
      5.0282,  -22.9052;
      5.7330,   -6.1621;
      4.6625,   -5.1385;
     29.1699,  -13.4408;
     25.2282,  -28.8829;
     14.9255,  -10.0067;
     20.6541,  -32.1381;
     18.9759,  -12.1868;
     12.7825,  -22.0223;
     34.2217,  -21.2411;
     23.1080,  -28.8980;
     23.5228,   -9.8477;
     35.2283,  -50.7007;
     46.1312,  -40.8638
     37.5468,  -58.2213;
     53.7743,  -42.0099;
     35.0350,  -45.8662;
     53.9225,  -25.8724;
     54.3669,  -62.3214;
     48.5339,  -33.5259;
     33.4506,  -60.9903;
     66.7284,  -50.3293;
     21.3952,  -41.1583;
     12.0079,   11.9377];
     
disp(solve_all(damper_x, wt));
u(5) = y;

# # # # # 

n = 96;
k = 48;
hx = l / n;
ht = t_total / k;
wt = zeros(k + 1, num_of_dampers);

alpha = osc_alpha();
beta = osc_beta();

b = osc_b();
c = osc_c();
c_inverted = inv(c);
sweep_side = eye(2);
cTilde = osc_cTilde();

y = osc_y();
y_sec = osc_sec_y();

xr = 0.0 : hx : l;
yr = 0.0 : ht : t_total;

eta = power(10, -3);
xi = power(10, -4);

 wt = [-1.485522,   -1.694532;
     -6.424197,   14.133831;
    -10.101980,   17.829227;
     -8.832759,   11.946591;
    -27.076803,    2.907637;
    -16.365961,   10.497619;
    -13.237924,   24.301485;
    -16.813525,   18.249961;
    -14.458377,   -0.420208;
     -9.405585,   29.466794;
     -4.417095,   16.700421;
    -10.375309,   -2.066320;
    -21.768101,    5.494118;
      2.043857,    4.948618;
     -0.036005,   -4.324727;
      3.746184,   -4.959997;
     11.893897,  -11.100032;
     22.750772,  -11.016368;
     21.929546,  -16.386440;
     17.604164,   -8.418723;
     16.582951,  -23.981786;
     18.141936,  -23.991904;
     13.231099,  -17.024127;
     27.689022,  -22.787353;
     18.710283,  -29.978201;
     13.342612,  -19.133219;
     15.711358,  -14.238627;
     24.818219,  -14.737150;
     25.380604,  -31.229509;
     38.409710,  -31.170090;
     38.884686,  -24.577345;
     35.073493,  -32.731381;
     20.480097,  -44.825859;
     40.366269,  -41.712957;
     53.308381,  -42.887457
     44.829572,  -47.358731;
     45.760230,  -58.580521;
     60.146907,  -41.743196;
     48.872428,  -35.716147;
     42.414037,  -44.935717;
     36.863223,  -44.639648;
     50.741697,  -57.017966;
     34.872610,  -52.122649;
     47.706788,  -42.942536;
     49.594645,  -41.212174;
     35.833930,  -35.033309;
     50.529841,  -54.817278;
     33.979975,  -29.491043;
      6.713462,   -3.832805];
     
disp(solve_all(damper_x, wt));
u(6) = y;

plot_u(u);

exit;

# print_task_init();

%plot_start_h(1);
%exit;

# wt = marquardt(@solve_all, damper_x, wt, Inf, vareps, power(10, 4));




# global hx = l / n;
# global ht = t_total / k;
# global y = osc_y();
# global y_sec = osc_sec_y();


#     
# disp(solve_all(damper_x, wt));
# u(2) = y;





%  

%  

print_dampers_info();
disp('E(T) = ');
err = solve_all(damper_x, wt);
disp(err);
disp('');

print_u_grid_values();

# plot_u_grid(err);
#plot_f(1, damper_x, wt);
%plot_wt(wt, 1);
