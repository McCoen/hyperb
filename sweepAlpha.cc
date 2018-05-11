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

cl_context sweepAlphaContext = NULL;
cl_program sweepAlphaProgram = NULL;
cl_kernel sweepAlphaKernel = NULL;

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

	command_queue = clCreateCommandQueueWithProperties(sweepAlphaContext, device_id, 0, &ret);

	clEnqueueNDRangeKernel(command_queue, sweepAlphaKernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFlush(command_queue);
	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, cl_sweep_alpha, CL_TRUE, 0, sizeof(cl_matrix2x2) * (n - 2), sweepAlphaArr, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	return sweepAlphaArr;
}

void releaseSweepAlphaKernel() {
	clReleaseProgram(sweepAlphaProgram);
	clReleaseKernel(sweepAlphaKernel);
	clReleaseContext(sweepAlphaContext);
}
