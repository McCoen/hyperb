//Compile with mkoctfile gpuMatrixV.cc -L"C:\hyperb" -lOpenCL -pedantic -Wall

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <octave-4.2.1/octave/oct.h>
#include <octave-4.2.1/octave/parse.h>

#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#include "hyperb.h"
#include <math.h>
#include <unistd.h>

cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_command_queue command_queue = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;

cl_context sweepAlphaContext = NULL;
cl_program sweepAlphaProgram = NULL;
cl_kernel sweepAlphaKernel = NULL;

cl_context context = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;

bool isConstantControl = false;

int n, k, numOfDampers;
cl_double a, l, t;

using namespace std;

void createAndBuildKernel() {
	FILE* fp;
	char* filename;
	if (isConstantControl) {
		filename = "constTridiag.cl";
	} else {
		filename = "tridiag.cl";
	}
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

	char gridSize[1024];
	sprintf(gridSize, "-DN=%d -DK=%d -DNUMBER_OF_DAMPERS=%d", n, k, numOfDampers);
	ret = clBuildProgram(program, 1, &device_id, gridSize, NULL, NULL);
	kernel = clCreateKernel(program, "energyInt", &ret);

	free(source_str);
}

void createAndBuildSweepAlphaKernel() {
	FILE* fp;
	const char* filename = "sweepAlpha.cl";
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

	sweepAlphaContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	sweepAlphaProgram = clCreateProgramWithSource(sweepAlphaContext, 1, (const char**) &source_str, (const size_t*) &source_size, &ret);

	char gridSize[1024];
	sprintf(gridSize, "-DN=%d -DK=%d -DNUMBER_OF_DAMPERS=%d", n, k, numOfDampers);
	ret = clBuildProgram(sweepAlphaProgram, 1, &device_id, gridSize, NULL, NULL);
	sweepAlphaKernel = clCreateKernel(sweepAlphaProgram, "sweepAlpha", &ret);

	free(source_str);
}

double* toDouble(Matrix u, int n, int k) {
	double* du = (double*) malloc(sizeof(double) * (n + 1) * (k + 1));
	register int i, j;
	for (i = 0; i <= k; i++) {
		for (j = 0; j <= n; j++) {
			du[j + i * (n + 1)] = u(i, j);
		}
	}
	return du;
}

void printGrid(double* u, int n, int k) {
	for (register int i = 0; i < k + 1; i++) {
		for (register int j = 0; j < n + 1; j++) {
			printf("%lf ", u[j + i * (n + 1)]);
		}
		puts("");
	}
}

void releaseAll() {
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseContext(context);
}

cl_matrix2x2* solveAlpha(cl_double* damperX, cl_double* wt) {
	size_t block = 1;
	size_t global_work_size[] = {block, 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_mem cl_sweep_alpha = clCreateBuffer(sweepAlphaContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x2) * (n - 2), NULL, NULL);

	clSetKernelArg(sweepAlphaKernel, 0, sizeof(cl_double), &a);
	clSetKernelArg(sweepAlphaKernel, 1, sizeof(cl_double), &l);
	clSetKernelArg(sweepAlphaKernel, 2, sizeof(cl_double), &t);
	clSetKernelArg(sweepAlphaKernel, 3, sizeof(cl_mem), &cl_sweep_alpha);

	cl_matrix2x2* sweepAlphaArr = (cl_matrix2x2*) malloc(sizeof(cl_matrix2x2) * (n - 2));

	command_queue = clCreateCommandQueue(sweepAlphaContext, device_id, 0, &ret);

	clEnqueueNDRangeKernel(command_queue, sweepAlphaKernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFlush(command_queue);
	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, cl_sweep_alpha, CL_TRUE, 0, sizeof(cl_matrix2x2) * (n - 2), sweepAlphaArr, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	clReleaseMemObject(cl_sweep_alpha);

	return sweepAlphaArr;
}

cl_double* solveMatrixV(cl_matrix2x2* sweepAlphaArr, cl_double* damperX, cl_double* wt, Matrix st, cl_double* u, cl_double* v) {
	size_t block = 1;
	size_t global_work_size[] = {block, 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_matrix2x1* cmbounds = (cl_matrix2x1*) malloc(sizeof(cl_matrix2x1) * numOfDampers);
	cl_matrix2x1* cmst = (cl_matrix2x1*) malloc(sizeof(cl_matrix2x1) * numOfDampers);
	for (register int i = 0; i < numOfDampers; i++) {
		cmbounds[i].a = NAN;
		cmbounds[i].b = NAN;
		cmst[i].a = st(i, 0);
		cmst[i].b = st(i, 1);
	}

	cl_mem cl_sweep_alpha = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_matrix2x2) * (n - 2), NULL, NULL);
	cl_mem cl_energy_int = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, NULL);
	cl_mem cl_damper_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers, NULL, NULL);
	cl_mem cl_wt = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);
	cl_mem cl_wt_bounds = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	cl_mem cl_st = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	cl_mem cl_u = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);
	cl_mem cl_v = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_sweep_alpha);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_energy_int);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_wt);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_wt_bounds);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_st);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_damper_x);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_u);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), &cl_v);
	clSetKernelArg(kernel, 8, sizeof(cl_double), &a);
	clSetKernelArg(kernel, 9, sizeof(cl_double), &l);
	clSetKernelArg(kernel, 10, sizeof(cl_double), &t);

	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	clEnqueueWriteBuffer(command_queue, cl_sweep_alpha, CL_TRUE, 0, sizeof(cl_matrix2x2) * (n - 2), sweepAlphaArr, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_wt, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * (k + 1), wt, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_wt_bounds, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmbounds, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_st, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmst, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_damper_x, CL_TRUE, 0, sizeof(cl_double) * numOfDampers, damperX, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1), u, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_v, CL_TRUE, 0, sizeof(cl_double) * (n + 1), v, 0, NULL, NULL);

	clEnqueueNDRangeKernel(command_queue, kernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFlush(command_queue);
	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, cl_v, CL_TRUE, 0, sizeof(cl_double) * (n + 1) * (k + 1), v, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	clReleaseMemObject(cl_sweep_alpha);
	clReleaseMemObject(cl_energy_int);
	clReleaseMemObject(cl_damper_x);
	clReleaseMemObject(cl_wt);
	clReleaseMemObject(cl_st);
	clReleaseMemObject(cl_u);
	clReleaseMemObject(cl_v);

	return v;
}

Matrix toMatrix(double* u, const int n, const int k) {
	Matrix mu(k + 1, n + 1);
	register int i, j;
	for (i = 0; i <= k; i++) {
		for (j = 0; j <= n; j++) {
			mu(i, j) = u[j + i * (n + 1)];
		}
	}
	return mu;
}

DEFUN_DLD(gpuMatrixV, args, nargout, "") {
	Matrix damperX = args(0).matrix_value();
	Matrix wt = args(1).matrix_value();
	Matrix st = args(2).matrix_value();

	numOfDampers = args(3).int_value();
	n = args(4).int_value();
	k = args(5).int_value();

	Matrix u = args(6).matrix_value();
	Matrix v = args(7).matrix_value();

	a = args(8).double_value();
	l = args(9).double_value();
	t = args(10).double_value();

	cl_double* du = toDouble(u, n, k);
	cl_double* dv = toDouble(v, n, k);

	cl_double* dampx = (cl_double*) malloc(sizeof(cl_double) * numOfDampers);
	for (register int i = 0; i < numOfDampers; i++) {
		dampx[i] = damperX(i);
	}

	for (register int i = 0; i < numOfDampers; i++) {
		if (isnan(dampx[i])) {
			isConstantControl = true;
			break;
		}
	}

	createAndBuildSweepAlphaKernel();
	createAndBuildKernel();

	cl_double* dwt = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * (k + 1));
	cl_double* dst = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * (k + 1));
	for (register int i = 0; i < numOfDampers; i++) {
		for (register int j = 0; j < k + 1; j++) {
			dwt[j + i * (k + 1)] = wt(j, i);
			dst[j + i * (k + 1)] = st(j, i);
		}
	}

	cl_matrix2x2* sweepAlphaArr = solveAlpha(dampx, dwt);
	dv = solveMatrixV(sweepAlphaArr, dampx, dwt, st, du, dv);

	v = toMatrix(dv, n, k);

	free(du);
	free(dv);
	free(dampx);
	free(dwt);
	free(dst);
	free(sweepAlphaArr);
	releaseAll();

	return octave_value(v);
}

