#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define USE_GPU
#define v403

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
cl_command_queue command_queue = NULL;
cl_mem memobj = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;

cl_context context = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;

void createAndBuildKernel() {
	FILE* fp;
	const char* filename = "grid.cl";
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

	puts("Now building kernel");

	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "solveGrid", &ret);

	puts("OpenCL kernel has built");
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

cl_double* solveGrid(cl_double* damperX, cl_double* wt, cl_double* st, cl_double* u, int n, int k, cl_double a, cl_double l, cl_double t) {
	size_t block = 1;
	size_t global_work_size[] = {block, 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_int* no_fault = (cl_int*) malloc(sizeof(cl_int));
	no_fault[0] = 0;

	int numOfDampers = 1;

	cl_mem cl_energy_int = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, NULL);
	cl_mem cl_damper_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers, NULL, NULL);
	cl_mem cl_wt = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);
	cl_mem cl_st = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);
	cl_mem cl_u = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);
	cl_mem cl_no_fault = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_energy_int);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_wt);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_st);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_damper_x);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_u);
	clSetKernelArg(kernel, 5, sizeof(cl_double), &a);
	clSetKernelArg(kernel, 6, sizeof(cl_double), &l);
	clSetKernelArg(kernel, 7, sizeof(cl_double), &t);
	clSetKernelArg(kernel, 8, sizeof(cl_mem), &cl_no_fault);

	cl_double* sei = (cl_double*) malloc(sizeof(cl_double));
	sei[0] = -1.0;

	do {
		command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

		clEnqueueWriteBuffer(command_queue, cl_energy_int, CL_TRUE, 0, sizeof(cl_double), sei, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_wt, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * (k + 1), wt, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_st, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * (k + 1), st, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_damper_x, CL_TRUE, 0, sizeof(cl_double) * numOfDampers, damperX, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1) * (k + 1), u, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_no_fault, CL_TRUE, 0, sizeof(cl_int), no_fault, 0, NULL, NULL);

		clEnqueueNDRangeKernel(command_queue, kernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

		clFlush(command_queue);
		clFinish(command_queue);

		clEnqueueReadBuffer(command_queue, cl_energy_int, CL_TRUE, 0, sizeof(cl_double), sei, 0, NULL, NULL);
		clEnqueueReadBuffer(command_queue, cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1) * (k + 1), u, 0, NULL, NULL);
		clEnqueueReadBuffer(command_queue, cl_no_fault, CL_TRUE, 0, sizeof(cl_int), no_fault, 0, NULL, NULL);

		clReleaseCommandQueue(command_queue);
	} while (no_fault[0] != 1);

	clReleaseMemObject(cl_energy_int);
	clReleaseMemObject(cl_damper_x);
	clReleaseMemObject(cl_wt);
	clReleaseMemObject(cl_st);
	clReleaseMemObject(cl_u);
	clReleaseMemObject(cl_no_fault);

	return u;
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

DEFUN_DLD(oscDeadening, args, nargout, "") {
	Matrix damperX = args(0).matrix_value();
	Matrix wt = args(1).matrix_value();
	Matrix st = args(2).matrix_value();

	int n = args(3).int_value();
	int k = args(4).int_value();
	Matrix u = args(5).matrix_value();

	cl_double a = args(6).double_value();
	cl_double l = args(7).double_value();
	cl_double t = args(8).double_value();

	createAndBuildKernel();
	
	double* du = toDouble(u, n, k);
	double dampx[]{damperX(0), damperX(1)};
	double dwt[k + 1];
	double dst[k + 1];
	for (register int i = 0; i < k + 1; i++) {
		dwt[i] = wt(i);
		dst[i] = st(i);
	}

	du = solveGrid(dampx, dwt, dst, du, n, k, a, l, t);
	u = toMatrix(du, n, k);

	/*
	output_file = fopen('marquardt_last', 'w');
	fprintf(output_file, "%.256e\n", mu);
	for i = 1 : length(wt(:))
		fprintf(output_file, "%.256e\n", wt_new(i));
	endfor
	fclose(output_file);
	*/

	releaseAll();
	return octave_value(u);
}
