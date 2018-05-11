#include <octave-4.2.1/octave/oct.h>
#include <octave-4.2.1/octave/parse.h>

#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#include "hyperb.h"

extern cl_platform_id platform_id;
extern cl_device_id device_id;
extern cl_command_queue command_queue;
extern cl_mem memobj;
extern cl_uint ret_num_devices;
extern cl_uint ret_num_platforms;
extern cl_int ret;

extern int n, k, numOfDampers;
extern cl_double a, l, t;

cl_context context = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;

void createAndBuildKernel() {
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

	char gridSize[1024];
	sprintf(gridSize, "-DN=%d -DK=%d -DNUMBER_OF_DAMPERS=%d", n, k, numOfDampers);
	ret = clBuildProgram(program, 1, &device_id, gridSize, NULL, NULL);
	kernel = clCreateKernel(program, "energyInt", &ret);

	free(source_str);
}

cl_double solveEnergyInt(cl_matrix2x2* sweepAlphaArr, cl_double* damperX, cl_double* wt, Matrix wtBounds, Matrix st, cl_double* u, cl_double* v, bool includePenalty) {
	size_t block = 1;
	size_t global_work_size[] = {block, 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_matrix2x1* cmbounds = (cl_matrix2x1*) malloc(sizeof(cl_matrix2x1) * numOfDampers);
	cl_matrix2x1* cmst = (cl_matrix2x1*) malloc(sizeof(cl_matrix2x1) * numOfDampers);
	for (register int i = 0; i < numOfDampers; i++) {
		if (includePenalty) {
			cmbounds[i].a = wtBounds(i, 0);
			cmbounds[i].b = wtBounds(i, 1);
		} else {
			cmbounds[i].a = NAN;
			cmbounds[i].b = NAN;
		}
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

	cl_double* sei = (cl_double*) malloc(sizeof(cl_double));

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

	clEnqueueReadBuffer(command_queue, cl_energy_int, CL_TRUE, 0, sizeof(cl_double), sei, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	clReleaseMemObject(cl_sweep_alpha);
	clReleaseMemObject(cl_energy_int);
	clReleaseMemObject(cl_damper_x);
	clReleaseMemObject(cl_wt);
	clReleaseMemObject(cl_st);
	clReleaseMemObject(cl_u);
	clReleaseMemObject(cl_v);

	return sei[0];

}

void releaseKernel() {
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseContext(context);
}
