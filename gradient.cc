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

cl_context gradientContext = NULL;
cl_program gradientProgram = NULL;
cl_kernel gradientKernel = NULL;

cl_mem grad_cl_f;
cl_mem grad_cl_gid;
cl_mem grad_cl_sweep_alpha;
cl_mem grad_cl_gradient;
cl_mem grad_cl_wt;
cl_mem grad_cl_st;
cl_mem grad_cl_damper_x;
cl_mem grad_cl_wt_bounds;
cl_mem grad_cl_u;
cl_mem grad_cl_v;

void createAndBuildGradientKernel() {
	const char* filename = "gradient.cl";
	size_t source_size;
	char* source_str;

	FILE* fp = fopen(filename, "r");

	if (!fp) {
		fprintf(stderr, "Fail\n");
		exit(0);
	}

	source_str = (char*) malloc(sizeof(char) * 100000);
	source_size = fread(source_str, 1, 100000, fp);
	fclose(fp);

	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	gradientContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	gradientProgram = clCreateProgramWithSource(gradientContext, 1, (const char**) &source_str, (const size_t*) &source_size, &ret);

	char gridSize[1024];
	sprintf(gridSize, "-DN=%d -DK=%d -DNUMBER_OF_DAMPERS=%d", n, k, numOfDampers);
	ret = clBuildProgram(gradientProgram, 1, &device_id, gridSize, NULL, NULL);
	gradientKernel = clCreateKernel(gradientProgram, "gradientAt", &ret);

	grad_cl_f = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1), NULL, NULL);
	grad_cl_gid = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, NULL);
	grad_cl_sweep_alpha = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x2) * (n - 2), NULL, NULL);
	grad_cl_gradient = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);
	grad_cl_wt = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);
	grad_cl_wt_bounds = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	grad_cl_st = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	grad_cl_damper_x = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers, NULL, NULL);
	grad_cl_u = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);
	grad_cl_v = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1), NULL, NULL);

	clSetKernelArg(gradientKernel, 0, sizeof(cl_mem), &grad_cl_f);
	clSetKernelArg(gradientKernel, 1, sizeof(cl_mem), &grad_cl_sweep_alpha);
	clSetKernelArg(gradientKernel, 3, sizeof(cl_mem), &grad_cl_gradient);
	clSetKernelArg(gradientKernel, 4, sizeof(cl_mem), &grad_cl_wt);
	clSetKernelArg(gradientKernel, 5, sizeof(cl_mem), &grad_cl_wt_bounds);
	clSetKernelArg(gradientKernel, 6, sizeof(cl_mem), &grad_cl_st);
	clSetKernelArg(gradientKernel, 7, sizeof(cl_mem), &grad_cl_damper_x);
	clSetKernelArg(gradientKernel, 8, sizeof(cl_mem), &grad_cl_u);
	clSetKernelArg(gradientKernel, 9, sizeof(cl_mem), &grad_cl_v);
	clSetKernelArg(gradientKernel, 10, sizeof(cl_double), &a);
	clSetKernelArg(gradientKernel, 11, sizeof(cl_double), &l);
	clSetKernelArg(gradientKernel, 12, sizeof(cl_double), &t);

	free(source_str);
}

Matrix gpuGradientAt(cl_matrix2x2* sweepAlphaArr, cl_double sei, Matrix wt, Matrix st, double derH, cl_double* dampx, Matrix wtBounds, cl_double* u, cl_double* v) {
	size_t block = k + 1;
	size_t global_work_size[] = {(size_t) (numOfDampers * (k + 1)), 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_double* grad = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * (k + 1));
	cl_double* dwt = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * (k + 1));
	for (register int i = 0; i < numOfDampers; i++) {
		for (register int j = 0; j < k + 1; j++) {
			dwt[j + i * (k + 1)] = wt(j, i);
		}
	}

	cl_matrix2x1* cmbounds = (cl_matrix2x1*) malloc(sizeof(cl_matrix2x1) * numOfDampers);
	cl_matrix2x1* cmst = (cl_matrix2x1*) malloc(sizeof(cl_matrix2x1) * numOfDampers);
	for (register int i = 0; i < numOfDampers; i++) {
		cmbounds[i].a = wtBounds(i, 0);
		cmbounds[i].b = wtBounds(i, 1);
		cmst[i].a = st(i, 0);
		cmst[i].b = st(i, 1);
	}

	Matrix gradient(numOfDampers * (k + 1), 1);

	clSetKernelArg(gradientKernel, 2, sizeof(cl_double), &sei);

	cl_command_queue command_queue = clCreateCommandQueueWithProperties(gradientContext, device_id, 0, &ret);

	clEnqueueWriteBuffer(command_queue, grad_cl_sweep_alpha, CL_TRUE, 0, sizeof(cl_matrix2x2) * (n - 2), sweepAlphaArr, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_wt_bounds, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmbounds, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_wt, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * (k + 1), dwt, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_st, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmst, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_damper_x, CL_TRUE, 0, sizeof(cl_double) * numOfDampers, dampx, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1), u, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_v, CL_TRUE, 0, sizeof(cl_double) * (n + 1), v, 0, NULL, NULL);

	clEnqueueNDRangeKernel(command_queue, gradientKernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFlush(command_queue);
	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, grad_cl_gradient, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * (k + 1), grad, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	for (register int i = 0; i < numOfDampers * (k + 1); i++) {
		gradient(i) = grad[i];
	}

	free(grad);
	free(dwt);

	return gradient;
}

void releaseGradientKernel() {
	clReleaseProgram(gradientProgram);
	clReleaseKernel(gradientKernel);
	clReleaseContext(gradientContext);
}

void printGradient(Matrix gradient, int iter, int m) {
	printf("Iteration %d/%d gradient:\n", iter, m);
	for (register int i = 0; i < k + 1; i++) {
		for (register int j = 0; j < numOfDampers; j++) {
			printf("%e\t", gradient(i + j * (k + 1)));
		}
		puts("");
	}
	printf("\n");
}

double euclideanGradientNorm(Matrix gradient, int k) {
	double n = 0.0;
	for (register int i = 0; i < numOfDampers * (k + 1); i++) {
		n += pow(gradient(i), 2);
	}
	return sqrt(n);
}
