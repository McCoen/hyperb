#include <octave-4.2.1/octave/oct.h>
#include <octave-4.2.1/octave/parse.h>

#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#include "hyperb.h"
#include <math.h>
#include <unistd.h>

extern cl_platform_id platform_id;
extern cl_device_id device_id;
extern cl_command_queue command_queue;
extern cl_mem memobj;
extern cl_uint ret_num_devices;
extern cl_uint ret_num_platforms;
extern cl_int ret;

extern int n, k, numOfDampers;
extern cl_double a, l, t;

extern cl_context context;
extern cl_program program;
extern cl_kernel kernel;

using namespace std;

void releaseAll() {
	releaseSweepAlphaKernel();
	releaseKernel();
}

cl_double* solveGrid(cl_matrix2x2* sweepAlphaArr, cl_double* damperX, cl_double* wt, Matrix st, cl_double* u, cl_double* v) {
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

	command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

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

	clEnqueueReadBuffer(command_queue, cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1) * (k + 1), u, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	clReleaseMemObject(cl_sweep_alpha);
	clReleaseMemObject(cl_energy_int);
	clReleaseMemObject(cl_damper_x);
	clReleaseMemObject(cl_wt);
	clReleaseMemObject(cl_st);
	clReleaseMemObject(cl_u);
	clReleaseMemObject(cl_v);

	return u;
}

DEFUN_DLD(gpuOscDeadening, args, nargout, "") {
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
			//isConstantControl = true;
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
	du = solveGrid(sweepAlphaArr, dampx, dwt, st, du, dv);

	//printGrid(du, n, k);
	u = toMatrix(du, n, k);

	free(du);
	free(dv);
	free(dampx);
	free(dwt);
	free(dst);
	free(sweepAlphaArr);
	releaseAll();

	return octave_value(u);
}

