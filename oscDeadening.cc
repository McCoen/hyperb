//#define USE_GPU
//#define v403

#ifdef v403
#include <octave-4.0.3/octave/oct.h>
#include <octave-4.0.3/octave/parse.h>
#else
#include <octave-3.8.2/octave/oct.h>
#include <octave-3.8.2/octave/parse.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include "hyperb.h"
#include <math.h>
#include <unistd.h>

cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_mem memobj = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;

double* toDouble(Matrix matU, const int n, const int k) {
    double* u = (double*) malloc(sizeof(double) * (k + 1) * (n + 1));
    for (register int i = 0; i <= k; i++) {
        for (register int j = 0; j <= n; j++) {
            u[j + i * (n + 1)] = matU(i, j);
        }
    }
    return u;
}

cl_matrix2x1 matrixMultiply(cl_matrix2x2 a, cl_matrix2x1 b) {
    cl_matrix2x1 c;
    c.a = a.a * b.a + a.b * b.b;
    c.b = a.c * b.a + a.d * b.b;
    return c;
}

cl_matrix2x2 matrixMultiply2x2(cl_matrix2x2 a, cl_matrix2x2 b) {
    cl_matrix2x2 c;
    c.a = a.a * b.a + a.b * b.c;
    c.b = a.a * b.b + a.b * b.d;
    c.c = a.c * b.a + a.d * b.c;
    c.d = a.c * b.b + a.d * b.d;
    return c;
}

cl_matrix2x2 matrixInverse(cl_matrix2x2 a) {
    double detDenominator = a.a * a.d - a.b * a.c;
    cl_matrix2x2 b;
    b.a = a.d / detDenominator;
    b.b = -a.b / detDenominator;
    b.c = -a.c / detDenominator;
    b.d = a.a / detDenominator;
    return b;
}

cl_matrix2x1 sweepBeta(cl_matrix2x2* sweepAlphaArr, cl_matrix2x1* sweepBetaArr, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 sweepSide, cl_matrix2x1* right, int i) {
    cl_matrix2x1 beta;
    cl_matrix2x1 f = right[i];
    if (i == 0) {
        beta = matrixMultiply(cInverted, f);
    } else {
        cl_matrix2x2 temp = c;
        cl_matrix2x2 temp2 = matrixMultiply2x2(sweepSide, sweepAlphaArr[i - 1]);
        temp.a -= temp2.a;
        temp.b -= temp2.b;
        temp.c -= temp2.c;
        temp.d -= temp2.d;

        cl_matrix2x2 invTemp = matrixInverse(temp);

        cl_matrix2x1 temp3 = matrixMultiply(sweepSide, sweepBetaArr[i - 1]);
        temp3.a += f.a;
        temp3.b += f.b;

        beta = matrixMultiply(invTemp, temp3);
    }
    return beta;
}

cl_matrix2x2 sweepAlpha(cl_matrix2x2* sweepAlphaArr, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 sweepSide, int i) {
    cl_matrix2x2 alpha;
    if (i == 0) {
        alpha = matrixMultiply2x2(cInverted, sweepSide);
    } else {
        cl_matrix2x2 temp = c;
        cl_matrix2x2 temp2 = matrixMultiply2x2(sweepSide, sweepAlphaArr[i - 1]);
        temp.a -= temp2.a;
        temp.b -= temp2.b;
        temp.c -= temp2.c;
        temp.d -= temp2.d;

        cl_matrix2x2 invTemp = matrixInverse(temp);

        alpha = matrixMultiply2x2(invTemp, sweepSide);
    }
    return alpha;
}

cl_matrix2x1* tridiagMatrixSweep(cl_matrix2x2* sweepAlphaArr, cl_matrix2x1* right, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 sweepSide, int n) {
    int sweepN = n - 1;
    cl_matrix2x1* x = new cl_matrix2x1[sweepN];

    register int i;
    cl_matrix2x1* sweepBetaArr = new cl_matrix2x1[sweepN];
    for (i = 0; i < sweepN; i++) {
        sweepBetaArr[i] = sweepBeta(sweepAlphaArr, sweepBetaArr, c, cInverted, sweepSide, right, i);
    }

    x[sweepN - 1] = sweepBetaArr[sweepN - 1];
    for (i = sweepN - 2; i >= 0; i--) {
        cl_matrix2x1 temp = matrixMultiply(sweepAlphaArr[i], x[i + 1]);
        temp.a += sweepBetaArr[i].a;
        temp.b += sweepBetaArr[i].b;
        x[i] = temp;
    }

    delete[] sweepBetaArr;
    return x;
}

double oscF(Matrix damperX, Matrix wt, int n, int m, double hx) {
    int numOfDampers = damperX.length();

    double x = hx * m;
    double f = 0.0;

    double a = 1.0;
    double l = 1.0;

    register int i;
    for (i = 0; i < numOfDampers; i++) {
        double currentWt = wt(n, i);

        double fNew = 0.0;

        if (x < damperX(i)) {
            fNew = -x / (a * l) * (l - damperX(i));
        } else {
            fNew = 1 / a * (x - damperX(i)) - x / (a * l) * (l - damperX(i));
        }

        f += fNew * currentWt;
    }
    return f;
}

cl_matrix2x1 oscV(Matrix damperX, Matrix wt, int m, int n, double hx) {
    cl_matrix2x1 v;
    v.a = -oscF(damperX, wt, m, n, hx) - oscF(damperX, wt, m + 1, n, hx);
    v.b = 0.0;
    return v;
}

cl_matrix2x1* oscRight(Matrix damperX, Matrix wt, Matrix u, Matrix v, int n, int m, double hx, cl_matrix2x2 cTilde, double beta) {
    cl_matrix2x1* right = (cl_matrix2x1*) malloc(sizeof(cl_matrix2x1) * (n - 1));

    register int i;
    for (i = 1; i < n; i++) {

        cl_matrix2x1 y1;
        y1.a = u(m, i);
        y1.b = v(m, i);

        cl_matrix2x1 temp1 = matrixMultiply(cTilde, y1);

        cl_matrix2x1 y0;
        y0.a = u(m, i - 1);
        y0.b = v(m, i - 1);

        cl_matrix2x1 temp = y0;
        temp.a -= temp1.a;
        temp.b -= temp1.b;

        cl_matrix2x1 y2;
        y2.a = u(m, i + 1);
        y2.b = v(m, i + 1);

        temp.a += y2.a;
        temp.b += y2.b;

        cl_matrix2x1 currentV = oscV(damperX, wt, m, i, hx);
        currentV.a *= beta;
        currentV.b *= beta;

        temp.a += currentV.a;
        temp.b += currentV.b;

        right[i - 1] = temp;
    }
    return right;
}

Matrix solveTask(Matrix damperX, Matrix wt, Matrix u, Matrix v, int n, int k, double hx, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 sweepSide, cl_matrix2x2 cTilde, double beta) {
#ifdef USE_GPU
	size_t block = 1;
	size_t global_work_size[] = {block, 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_mem cl_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * (n - 1), NULL, NULL);
	cl_mem cl_right = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_matrix2x1) * (n - 1), NULL, NULL);
	cl_mem cl_c = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_matrix2x2), NULL, NULL);
	cl_mem cl_c_inverted = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_matrix2x2), NULL, NULL);
	cl_mem cl_sweep_side = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_matrix2x2), NULL, NULL);
	cl_mem cl_no_fault = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int), NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_x);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_right);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_c);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_c_inverted);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_sweep_side);
	clSetKernelArg(kernel, 5, sizeof(cl_int), &n);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_no_fault);

    register int i, j;
    for (i = 0; i < k; i++) {
	cl_matrix2x1* right = oscRight(damperX, wt, u, v, n, i, hx, cTilde, beta);

	cl_matrix2x1* x = (cl_matrix2x1*) malloc(sizeof(cl_matrix2x1) * (n - 1));
	cl_int* no_fault = (cl_int*) malloc(sizeof(cl_int));
	no_fault[0] = 0;

	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	clEnqueueWriteBuffer(command_queue, cl_x, CL_TRUE, 0, sizeof(cl_matrix2x1) * (n - 1), x, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_right, CL_TRUE, 0, sizeof(cl_matrix2x1) * (n - 1), right, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_c, CL_TRUE, 0, sizeof(cl_matrix2x2), &c, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_c_inverted, CL_TRUE, 0, sizeof(cl_matrix2x2), &cInverted, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_sweep_side, CL_TRUE, 0, sizeof(cl_matrix2x2), &sweepSide, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_no_fault, CL_TRUE, 0, sizeof(cl_int), no_fault, 0, NULL, NULL);

	ret = clEnqueueNDRangeKernel(command_queue, kernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFlush(command_queue);
	clFinish(command_queue);

	ret = clEnqueueReadBuffer(command_queue, cl_x, CL_TRUE, 0, sizeof(cl_matrix2x1) * (n - 1), x, 0, NULL, NULL);
	ret = clEnqueueReadBuffer(command_queue, cl_no_fault, CL_TRUE, 0, sizeof(cl_int), no_fault, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	if (no_fault[0] == 0) {
		i--;
		continue;
	}

        //delete[] right;

        for (j = 0; j < n - 1; j++) {
            cl_matrix2x1 currentX = x[j];
            u(i + 1, j + 1) = currentX.a;
            v(i + 1, j + 1) = currentX.b;
        }
        //delete[] x;
    }

    return u;
#else
    register int i, j;
    cl_matrix2x2* sweepAlphaArr = new cl_matrix2x2[n - 2];
    for (i = 0; i < n - 2; i++) {
        sweepAlphaArr[i] = sweepAlpha(sweepAlphaArr, c, cInverted, sweepSide, i);
    }
    for (i = 0; i < k; i++) {
        cl_matrix2x1* right = oscRight(damperX, wt, u, v, n, i, hx, cTilde, beta);
        cl_matrix2x1* x = tridiagMatrixSweep(sweepAlphaArr, right, c, cInverted, sweepSide, n);

        for (j = 0; j < n - 1; j++) {
            cl_matrix2x1 currentX = x[j];
            u(i + 1, j + 1) = currentX.a;
            v(i + 1, j + 1) = currentX.b;
        }

        free(right);
        delete[] x;
    }
    delete[] sweepAlphaArr;

    return u;
#endif
}


void printGrid(Matrix u, int n, int k) {
    register int i, j;
    for (i = 0; i < k + 1; i++) {
        for (j = 0; j < n + 1; j++) {
            printf("%.16e ", u(i, j));
        }
        puts("");
    }
}

double u_t(double* u, const int n, const int k, double ht, int nx) {
    return (u[nx + (k - 2) * (n + 1)] - 4 * u[nx + (k - 1) * (n + 1)] + 3 * u[nx + k * (n + 1)]) / (2 * ht);
}

double u_x(double* u, const int n, const int k, double hx, int nx) {
    if (nx == 0) {
        return (-3 * u[k * (n + 1)] + 4 * u[1 + k * (n + 1)] - u[2 + k * (n + 1)]) / (2 * hx);
    } else if (nx == n) {
        return (u[(n - 2) + k * (n + 1)] - 4 * u[(n - 1) + k * (n + 1)] + 3 * u[n + k * (n + 1)]) / (2 * hx);
    } else {
        double f2 = u[(nx + 1) + k * (n + 1)];
        double f0 = u[(nx - 1) + k * (n + 1)];
        return (f2 - f0) / (2 * hx);
    }
}

double u_xx(double* u, const int n, const int k, double hx, int nx) {
    if (nx == 0) {
        return (-3 * u[k * (n + 1)] + 4 * u[1 + k * (n + 1)] - u[2 + k * (n + 1)]) / (2 * hx);
    } else if (nx == n) {
        return (u[(n - 2) + k * (n + 1)] - 4 * u[(n - 1) + k * (n + 1)] + 3 * u[n + k * (n + 1)]) / (2 * hx);
    } else {
        return (u[(nx - 1) + k * (n + 1)] - 2 * u[nx + k * (n + 1)] + u[(nx + 1) + k * (n + 1)]) / pow(hx, 2);
    }
}

double simpsonInt(double* f, double hx, const int n) {
    double numInt = f[0];
    for (register int i = 1; i < n; i += 2) {
        numInt += 4 * f[i];
    }
    for (register int i = 2; i < n; i += 2) {
        numInt += 2 * f[i];
    }
    numInt += f[n];

    numInt *= hx / 3.0;
    return numInt;
}

double energyInt(double* u, const int n, const int k, double hx, double ht) {
    double a = 1.0, l = 1.0;
    double* f = (double*) malloc(sizeof(double) * (n + 1));
    for (register int nx = 0; nx <= n; nx++) {
        f[nx] = pow(u_t(u, n, k, ht, nx), 2) + pow(a, 4) * pow(u_xx(u, n, k, hx, nx), 2);
    }

    return simpsonInt(f, hx, n);

    /*octave_value_list in;
    in(0) = octave_value(trapzA);
    in(1) = octave_value(trapzB);

    octave_value_list out = feval("trapz", in, 1);
    return out(0).double_value();*/
}

double derivativeAt(Matrix damperX, Matrix wt, Matrix u, Matrix v, int n, int k, double hx, double ht, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 sweepSide, cl_matrix2x2 cTilde, double beta, int nth) {
    double derH = pow(10, -4);

    Matrix wt1 = wt;

    wt1(nth) = wt1(nth) + derH;

    Matrix u0 = solveTask(damperX, wt, u, v, n, k, hx, c, cInverted, sweepSide, cTilde, beta);
    Matrix u1 = solveTask(damperX, wt1, u, v, n, k, hx, c, cInverted, sweepSide, cTilde, beta);

    double* du0 = toDouble(u0, n, k);
    double* du1 = toDouble(u1, n, k);

    double f0 = energyInt(du0, n, k, hx, ht);
    double f1 = energyInt(du1, n, k, hx, ht);

	//printf("%.16e\n", f0);
	//printf("%.16e\n", f1);

    return (f1 - f0) / derH;
}

DEFUN_DLD(gpuHessianAt, args, nargout, "") {
    Matrix damperX = args(0).matrix_value();
    Matrix wt = args(1).matrix_value();
    int n = args(2).int_value();
    int k = args(3).int_value();
    double hx = args(4).double_value();
    double ht = args(5).double_value();

    Matrix u = args(6).matrix_value();
    Matrix v = args(7).matrix_value();
    Matrix matrixC = args(8).matrix_value();
    Matrix matrixInvertedC = args(9).matrix_value();
    Matrix matrixSweepSide = args(10).matrix_value();
    Matrix matrixTildeC = args(11).matrix_value();

    double beta = args(12).double_value();

    Matrix gradient = args(13).matrix_value();

    cl_matrix2x2 c;
    c.a = matrixC(0, 0);
    c.b = matrixC(0, 1);
    c.c = matrixC(1, 0);
    c.d = matrixC(1, 1);

    cl_matrix2x2 cInverted;
    cInverted.a = matrixInvertedC(0, 0);
    cInverted.b = matrixInvertedC(0, 1);
    cInverted.c = matrixInvertedC(1, 0);
    cInverted.d = matrixInvertedC(1, 1);

    cl_matrix2x2 sweepSide;
    sweepSide.a = matrixSweepSide(0, 0);
    sweepSide.b = matrixSweepSide(0, 1);
    sweepSide.c = matrixSweepSide(1, 0);
    sweepSide.d = matrixSweepSide(1, 1);

    cl_matrix2x2 cTilde;
    cTilde.a = matrixTildeC(0, 0);
    cTilde.b = matrixTildeC(0, 1);
    cTilde.c = matrixTildeC(1, 0);
    cTilde.d = matrixTildeC(1, 1);

#ifdef USE_GPU
    FILE* fp;
    const char* filename = "tridiag.cl";
    size_t source_size;
    char* source_str;

    fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Fail\n");
        exit(0);
    }

    source_str = (char*) malloc(sizeof(char) * 100000);
    source_size = fread(source_str, 1, 100000, fp);
    fclose(fp);

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    program = clCreateProgramWithSource(context, 1, (const char**) &source_str, (const size_t*) &source_size, &ret);

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "tridiag", &ret);
#endif

    double derH = pow(10, -4);
    register int i, j, m = wt.length();
    Matrix h(m, m);
    for (i =  0; i < m; i++) {
        double d0 = gradient(i);
        for (j = i; j < m; j++) {
            Matrix wtNew = wt;
            wtNew(j) += derH;
            double d1 = derivativeAt(damperX, wtNew, u, v, n, k, hx, ht, c, cInverted, sweepSide, cTilde, beta, i);
            double sd = (d1 - d0) / derH;
		//printf("%.16e\n", sd);
            h(i, j) = sd;
            h(j, i) = sd;
        }
    }

    return octave_value(h);
}

DEFUN_DLD(gpuGradientAt, args, nargout, "") {
    Matrix damperX = args(0).matrix_value();
    Matrix wt = args(1).matrix_value();
    int n = args(2).int_value();
    int k = args(3).int_value();
    double hx = args(4).double_value();
    double ht = args(5).double_value();

    Matrix u = args(6).matrix_value();
    Matrix v = args(7).matrix_value();
    Matrix matrixC = args(8).matrix_value();
    Matrix matrixInvertedC = args(9).matrix_value();
    Matrix matrixSweepSide = args(10).matrix_value();
    Matrix matrixTildeC = args(11).matrix_value();

    double beta = args(12).double_value();

    cl_matrix2x2 c;
    c.a = matrixC(0, 0);
    c.b = matrixC(0, 1);
    c.c = matrixC(1, 0);
    c.d = matrixC(1, 1);

    cl_matrix2x2 cInverted;
    cInverted.a = matrixInvertedC(0, 0);
    cInverted.b = matrixInvertedC(0, 1);
    cInverted.c = matrixInvertedC(1, 0);
    cInverted.d = matrixInvertedC(1, 1);

    cl_matrix2x2 sweepSide;
    sweepSide.a = matrixSweepSide(0, 0);
    sweepSide.b = matrixSweepSide(0, 1);
    sweepSide.c = matrixSweepSide(1, 0);
    sweepSide.d = matrixSweepSide(1, 1);

    cl_matrix2x2 cTilde;
    cTilde.a = matrixTildeC(0, 0);
    cTilde.b = matrixTildeC(0, 1);
    cTilde.c = matrixTildeC(1, 0);
    cTilde.d = matrixTildeC(1, 1);

#ifdef USE_GPU
    FILE* fp;
	const char* filename = "tridiag.cl";
	size_t source_size;
	char* source_str;

	fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Fail\n");
		exit(0);
	}

	source_str = (char*) malloc(sizeof(char) * 100000);
	source_size = fread(source_str, 1, 100000, fp);
	fclose(fp);

	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	program = clCreateProgramWithSource(context, 1, (const char**) &source_str, (const size_t*) &source_size, &ret);

	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "tridiag", &ret);
#endif

    u = solveTask(damperX, wt, u, v, n, k, hx, c, cInverted, sweepSide, cTilde, beta);
    double* du = toDouble(u, n, k);
	//printGrid(u, n, k);
    double ei = energyInt(du, n, k, hx, ht);

    double derH = pow(10, -4);
    int m = wt.length();
    Matrix gradient(m, 1);
    register int i;
    for (i = 0; i < m; i++) {
        Matrix wtNew = wt;
        wtNew(i) += derH;

        Matrix uNew = solveTask(damperX, wtNew, u, v, n, k, hx, c, cInverted, sweepSide, cTilde, beta);

        double* duNew = toDouble(uNew, n, k);
        double eiNew = energyInt(duNew, n, k, hx, ht);
        gradient(i) = (eiNew - ei) / derH;
    }

    return octave_value(gradient);
}

DEFUN_DLD(energyInt, args, nargout, "") {
    Matrix damperX = args(0).matrix_value();
    Matrix wt = args(1).matrix_value();
    int n = args(2).int_value();
    int k = args(3).int_value();
    double hx = args(4).double_value();
    double ht = args(5).double_value();

    Matrix u = args(6).matrix_value();
    Matrix v = args(7).matrix_value();
    Matrix matrixC = args(8).matrix_value();
    Matrix matrixInvertedC = args(9).matrix_value();
    Matrix matrixSweepSide = args(10).matrix_value();
    Matrix matrixTildeC = args(11).matrix_value();

    double beta = args(12).double_value();

    cl_matrix2x2 c;
    c.a = matrixC(0, 0);
    c.b = matrixC(0, 1);
    c.c = matrixC(1, 0);
    c.d = matrixC(1, 1);

    cl_matrix2x2 cInverted;
    cInverted.a = matrixInvertedC(0, 0);
    cInverted.b = matrixInvertedC(0, 1);
    cInverted.c = matrixInvertedC(1, 0);
    cInverted.d = matrixInvertedC(1, 1);

    cl_matrix2x2 sweepSide;
    sweepSide.a = matrixSweepSide(0, 0);
    sweepSide.b = matrixSweepSide(0, 1);
    sweepSide.c = matrixSweepSide(1, 0);
    sweepSide.d = matrixSweepSide(1, 1);

    cl_matrix2x2 cTilde;
    cTilde.a = matrixTildeC(0, 0);
    cTilde.b = matrixTildeC(0, 1);
    cTilde.c = matrixTildeC(1, 0);
    cTilde.d = matrixTildeC(1, 1);
#ifdef USE_GPU
    FILE* fp;
	const char* filename = "tridiag.cl";
	size_t source_size;
	char* source_str;

	fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Fail\n");
		exit(0);
	}

	source_str = (char*) malloc(sizeof(char) * 100000);
	source_size = fread(source_str, 1, 100000, fp);
	fclose(fp);

	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	program = clCreateProgramWithSource(context, 1, (const char**) &source_str, (const size_t*) &source_size, &ret);

	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "tridiag", &ret);
#endif

    u = solveTask(damperX, wt, u, v, n, k, hx, c, cInverted, sweepSide, cTilde, beta);
    double* du = toDouble(u, n, k);
    double sei = energyInt(du, n, k, hx, ht);
    return octave_value(sei);
}

DEFUN_DLD(oscDeadening, args, nargout, "Hello World Help String") {
    Matrix damperX = args(0).matrix_value();
    Matrix wt = args(1).matrix_value();
    int n = args(2).int_value();
    int k = args(3).int_value();
    double hx = args(4).double_value();
    double ht = args(5).double_value();

    Matrix u = args(6).matrix_value();
    Matrix v = args(7).matrix_value();
    Matrix matrixC = args(8).matrix_value();
    Matrix matrixInvertedC = args(9).matrix_value();
    Matrix matrixSweepSide = args(10).matrix_value();
    Matrix matrixTildeC = args(11).matrix_value();

    double beta = args(12).double_value();

    cl_matrix2x2 c;
    c.a = matrixC(0, 0);
    c.b = matrixC(0, 1);
    c.c = matrixC(1, 0);
    c.d = matrixC(1, 1);

    cl_matrix2x2 cInverted;
    cInverted.a = matrixInvertedC(0, 0);
    cInverted.b = matrixInvertedC(0, 1);
    cInverted.c = matrixInvertedC(1, 0);
    cInverted.d = matrixInvertedC(1, 1);

    cl_matrix2x2 sweepSide;
    sweepSide.a = matrixSweepSide(0, 0);
    sweepSide.b = matrixSweepSide(0, 1);
    sweepSide.c = matrixSweepSide(1, 0);
    sweepSide.d = matrixSweepSide(1, 1);

    cl_matrix2x2 cTilde;
    cTilde.a = matrixTildeC(0, 0);
    cTilde.b = matrixTildeC(0, 1);
    cTilde.c = matrixTildeC(1, 0);
    cTilde.d = matrixTildeC(1, 1);
#ifdef USE_GPU
    FILE* fp;
	const char* filename = "tridiag.cl";
	size_t source_size;
	char* source_str;

	fp = fopen(filename, "r");
	if (!fp) {
		fprintf(stderr, "Fail\n");
		exit(0);
	}

	source_str = (char*) malloc(sizeof(char) * 100000);
	source_size = fread(source_str, 1, 100000, fp);
	fclose(fp);

	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	program = clCreateProgramWithSource(context, 1, (const char**) &source_str, (const size_t*) &source_size, &ret);

	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "tridiag", &ret);
#endif

    u = solveTask(damperX, wt, u, v, n, k, hx, c, cInverted, sweepSide, cTilde, beta);

    return octave_value(u);
}
