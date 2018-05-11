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

cl_context hessianContext = NULL;
cl_program hessianProgram = NULL;
cl_kernel hessianKernel = NULL;

cl_mem hessian_cl_gid;
cl_mem hessian_cl_sweep_alpha;
cl_mem hessian_cl_gradient;
cl_mem hessian_cl_hessian;
cl_mem hessian_cl_wt;
cl_mem hessian_cl_st;
cl_mem hessian_cl_damper_x;
cl_mem hessian_cl_wt_bounds;
cl_mem hessian_cl_u;
cl_mem hessian_cl_v;

cl_mem cl_buffer;

void createAndBuildHessianKernel(cl_matrix2x2* sweepAlphaArr, Matrix wt, Matrix st, cl_double derH, cl_double* dampx, Matrix wtBounds, cl_double* du) {
	const char* filename = "hessian.cl";
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

	hessianContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	hessianProgram = clCreateProgramWithSource(hessianContext, 1, (const char**) &source_str, (const size_t*) &source_size, &ret);

	char gridSize[1024];
	sprintf(gridSize, "-DN=%d -DK=%d -DNUMBER_OF_DAMPERS=%d", n, k, numOfDampers);
	ret = clBuildProgram(hessianProgram, 1, &device_id, gridSize, NULL, NULL);
	hessianKernel = clCreateKernel(hessianProgram, "hessianAt", &ret);

	hessian_cl_gid = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, NULL);
	hessian_cl_sweep_alpha = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x2) * (n - 2), NULL, NULL);
	hessian_cl_gradient = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);
	hessian_cl_hessian = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (numOfDampers * (k + 1)) * (numOfDampers * (k + 1)), NULL, NULL);
	hessian_cl_wt = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);
	hessian_cl_wt_bounds = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	hessian_cl_st = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	hessian_cl_damper_x = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers, NULL, NULL);
	hessian_cl_u = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);
	hessian_cl_v = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1), NULL, NULL);

	cl_buffer = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);

	clSetKernelArg(hessianKernel, 0, sizeof(cl_mem), &hessian_cl_sweep_alpha);
	clSetKernelArg(hessianKernel, 1, sizeof(cl_mem), &hessian_cl_gradient);
	clSetKernelArg(hessianKernel, 2, sizeof(cl_mem), &hessian_cl_hessian);
	clSetKernelArg(hessianKernel, 3, sizeof(cl_mem), &hessian_cl_wt);
	clSetKernelArg(hessianKernel, 4, sizeof(cl_mem), &hessian_cl_wt_bounds);
	clSetKernelArg(hessianKernel, 5, sizeof(cl_mem), &hessian_cl_st);
	clSetKernelArg(hessianKernel, 6, sizeof(cl_mem), &hessian_cl_damper_x);
	clSetKernelArg(hessianKernel, 7, sizeof(cl_mem), &hessian_cl_u);
	clSetKernelArg(hessianKernel, 8, sizeof(cl_mem), &hessian_cl_v);
	clSetKernelArg(hessianKernel, 9, sizeof(cl_double), &a);
	clSetKernelArg(hessianKernel, 10, sizeof(cl_double), &l);
	clSetKernelArg(hessianKernel, 11, sizeof(cl_double), &t);
	clSetKernelArg(hessianKernel, 12, sizeof(cl_mem), &cl_buffer);

	free(source_str);
}

Matrix gpuHessianAt(cl_matrix2x2* sweepAlphaArr, Matrix gradient, Matrix wt, Matrix st, double derH, double* dampx, Matrix wtBounds, cl_double* u, cl_double* v, bool isDebug) {
	size_t block = k + 1;
	size_t global_work_size[] = {(size_t) ((numOfDampers * (k + 1)) * (numOfDampers * (k + 1))), 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_double* grad = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * (k + 1));
	cl_double* dwt = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * (k + 1));
	for (register int i = 0; i < numOfDampers; i++) {
		for (register int j = 0; j < k + 1; j++) {
			grad[j + i * (k + 1)] = gradient(j + i * (k + 1));
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

	Matrix hessian(numOfDampers * (k + 1), numOfDampers * (k + 1));
	cl_double* h = (cl_double*) malloc(sizeof(cl_double) * (numOfDampers * (k + 1)) * (numOfDampers * (k + 1)));

	cl_command_queue command_queue = clCreateCommandQueueWithProperties(hessianContext, device_id, 0, &ret);

	clEnqueueWriteBuffer(command_queue, hessian_cl_sweep_alpha, CL_TRUE, 0, sizeof(cl_matrix2x2) * (n - 2), sweepAlphaArr, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_wt_bounds, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmbounds, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_gradient, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * (k + 1), grad, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_wt, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * (k + 1), dwt, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_st, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmst, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_damper_x, CL_TRUE, 0, sizeof(cl_double) * numOfDampers, dampx, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1), u, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_v, CL_TRUE, 0, sizeof(cl_double) * (n + 1), v, 0, NULL, NULL);

	clEnqueueNDRangeKernel(command_queue, hessianKernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFlush(command_queue);
	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, hessian_cl_hessian, CL_TRUE, 0, sizeof(cl_double) * (numOfDampers * (k + 1)) * (numOfDampers * (k + 1)), h, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	for (register int i = 0; i < numOfDampers * (k + 1); i++) {
		for (register int j = 0; j < numOfDampers * (k + 1); j++) {
			hessian(i, j) = h[j + i * numOfDampers * (k + 1)];
		}
	}

	free(grad);
	free(h);
	free(dwt);

	return hessian;
}

void releaseHessianKernel() {
	clReleaseProgram(hessianProgram);
	clReleaseKernel(hessianKernel);
	clReleaseContext(hessianContext);
}

void printHessianMatrix(Matrix h, int iter, int m) {
	printf("Iteration %d/%d Hessian matrix:\n", iter, m);
	for (register int i = 0; i < numOfDampers * (k + 1); i++) {
		for (register int j = 0; j < numOfDampers * (k + 1); j++) {
			printf("%lf ", h(i, j));
		}
		puts("");
	}
	printf("\n");
}

double euclideanHessianNorm(Matrix h, int k) {
	double n = 0.0;
	for (register int i = 0; i < numOfDampers * (k + 1); i++) {
		for (register int j = 0; j < numOfDampers * (k + 1); j++) {
			n += pow(h(i, j), 2);
		}
	}
	return sqrt(n);
}
