//Compile with mkoctfile gpuEmpiricalApproach.cc -L"C:\hyperb" -lOpenCL -pedantic -Wall

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define USE_GPU
#define v403

#include <octave-4.2.1/octave/oct.h>
#include <octave-4.2.1/octave/parse.h>

#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"
#include "hyperb.h"
#include <math.h>
#include <unistd.h>

using namespace std;

cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_command_queue command_queue = NULL;
cl_mem memobj = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;

cl_context sweepAlphaContext = NULL;
cl_program sweepAlphaProgram = NULL;
cl_kernel sweepAlphaKernel = NULL;

cl_context context = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;

cl_context gradientContext = NULL;
cl_program gradientProgram = NULL;
cl_kernel gradientKernel = NULL;

cl_context hessianContext = NULL;
cl_program hessianProgram = NULL;
cl_kernel hessianKernel = NULL;

cl_context newhessianContext = NULL;
cl_program newhessianProgram = NULL;
cl_kernel newhessianKernel = NULL;

int n, k, numOfDampers;
cl_double a, l, t;

bool isConstantControl = false;
bool isPenaltyMinimization = false;

cl_mem grad_cl_f;
cl_mem grad_cl_gid;
cl_mem grad_cl_sweep_alpha;
cl_mem grad_cl_gradient;
cl_mem grad_cl_et;
cl_mem grad_cl_st;
cl_mem grad_cl_damper_x;
cl_mem grad_cl_wt_bounds;
cl_mem grad_cl_u;
cl_mem grad_cl_v;

cl_mem hessian_cl_gid;
cl_mem hessian_cl_sweep_alpha;
cl_mem hessian_cl_gradient;
cl_mem hessian_cl_hessian;
cl_mem hessian_cl_et;
cl_mem hessian_cl_st;
cl_mem hessian_cl_damper_x;
cl_mem hessian_cl_wt_bounds;
cl_mem hessian_cl_u;
cl_mem hessian_cl_v;

cl_mem cl_buffer;

using namespace std;

double hessianMaxAbs(Matrix h) {
	double maxAbs = 0.0;
	for (register int i = 0; i < numOfDampers * (k + 1); i++) {
		for (register int j = 0; j < numOfDampers * (k + 1); j++) {
			if (abs(h(i, j) > maxAbs)) {
				maxAbs = abs(h(i, j));
			}
		}
	}
	return maxAbs;
}

double euclideanEmpiricalHessianNorm(Matrix h, int k) {
	double n = 0.0;
	for (register int i = 0; i < numOfDampers * 7; i++) {
		for (register int j = 0; j < numOfDampers * 7; j++) {
			n += pow(h(i, j), 2);
		}
	}
	return sqrt(n);
}

void createAndBuildGradientKernel() {
	FILE* fp;
	size_t source_size;
	char* source_str;

	if (isConstantControl) {
		const char* filename = "constGradient.cl";
		fp = fopen(filename, "r");
	} else {
		const char* filename = "empGradient.cl";
		fp = fopen(filename, "r");
	}
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
	grad_cl_et = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * 7, NULL, NULL);
	grad_cl_wt_bounds = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	grad_cl_st = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	grad_cl_damper_x = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers, NULL, NULL);
	grad_cl_u = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);
	grad_cl_v = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1), NULL, NULL);

	clSetKernelArg(gradientKernel, 0, sizeof(cl_mem), &grad_cl_f);
	clSetKernelArg(gradientKernel, 1, sizeof(cl_mem), &grad_cl_sweep_alpha);
	clSetKernelArg(gradientKernel, 3, sizeof(cl_mem), &grad_cl_gradient);
	clSetKernelArg(gradientKernel, 4, sizeof(cl_mem), &grad_cl_et);
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

void createAndBuildKernel() {
	FILE* fp;
	char* filename;
	if (isConstantControl) {
		filename = "constTridiag.cl";
	} else {
		filename = "empTridiag.cl";
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

void printEmpiricalHessianMatrix(Matrix h, int iter, int m) {
	printf("Iteration %d/%d Hessian matrix:\n", iter, m);
	for (register int i = 0; i < numOfDampers * 7; i++) {
		for (register int j = 0; j < numOfDampers * 7; j++) {
			printf("%lf ", h(i, j));
		}
		puts("");
	}
	printf("\n");
}

void printEmpiricalGradient(Matrix gradient, int iter, int m) {
	printf("Iteration %d/%d gradient:\n", iter, m);
	for (register int i = 0; i < 7; i++) {
		for (register int j = 0; j < numOfDampers; j++) {
			printf("%e\t", gradient(i + j * 7));
		}
		puts("");
	}
	printf("\n");
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

	clReleaseProgram(gradientProgram);
	clReleaseKernel(gradientKernel);
	clReleaseContext(gradientContext);

	clReleaseProgram(hessianProgram);
	clReleaseKernel(hessianKernel);
	clReleaseContext(hessianContext);
}

cl_matrix2x2* solveAlpha(cl_double* damperX, cl_double* wt, cl_double* u) {
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

	return sweepAlphaArr;
}

cl_double solveEmpiricalEnergyInt(cl_matrix2x2* sweepAlphaArr, cl_double* damperX, cl_double* et, Matrix wtBounds, Matrix st, cl_double* u, cl_double* v, bool includePenalty) {
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
	cl_mem cl_et = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * 7, NULL, NULL);
	cl_mem cl_wt_bounds = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	cl_mem cl_st = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	cl_mem cl_u = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);
	cl_mem cl_v = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_sweep_alpha);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_energy_int);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_et);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_wt_bounds);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_st);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_damper_x);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), &cl_u);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), &cl_v);
	clSetKernelArg(kernel, 8, sizeof(cl_double), &a);
	clSetKernelArg(kernel, 9, sizeof(cl_double), &l);
	clSetKernelArg(kernel, 10, sizeof(cl_double), &t);

	cl_double* sei = (cl_double*) malloc(sizeof(cl_double));

	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	clEnqueueWriteBuffer(command_queue, cl_sweep_alpha, CL_TRUE, 0, sizeof(cl_matrix2x2) * (n - 2), sweepAlphaArr, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, cl_et, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * 7, et, 0, NULL, NULL);
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
	clReleaseMemObject(cl_et);
	clReleaseMemObject(cl_st);
	clReleaseMemObject(cl_u);
	clReleaseMemObject(cl_v);

	return sei[0];

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

Matrix gpuHessianAt(cl_matrix2x2* sweepAlphaArr, Matrix gradient, Matrix et, Matrix st, double derH, double* dampx, Matrix wtBounds, cl_double* u, cl_double* v, bool isDebug, cl_bool lockAmplitudeParams) {
	size_t block = numOfDampers * 7;
	size_t global_work_size[] = {block * block, 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_double* grad = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * 7);
	cl_double* det = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * 7);
	for (register int i = 0; i < numOfDampers; i++) {
		for (register int j = 0; j < 7; j++) {
			grad[j + i * 7] = gradient(j + i * 7);
			det[j + i * 7] = et(j, i);
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

	Matrix hessian(numOfDampers * 7, numOfDampers * 7);
	cl_double* h = (cl_double*) malloc(sizeof(cl_double) * (numOfDampers * 7) * (numOfDampers * 7));

	cl_command_queue command_queue = clCreateCommandQueue(hessianContext, device_id, 0, &ret);

	clEnqueueWriteBuffer(command_queue, hessian_cl_sweep_alpha, CL_TRUE, 0, sizeof(cl_matrix2x2) * (n - 2), sweepAlphaArr, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_wt_bounds, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmbounds, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_gradient, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * 7, grad, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_et, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * 7, det, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_st, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmst, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_damper_x, CL_TRUE, 0, sizeof(cl_double) * numOfDampers, dampx, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1), u, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, hessian_cl_v, CL_TRUE, 0, sizeof(cl_double) * (n + 1), v, 0, NULL, NULL);

	clEnqueueNDRangeKernel(command_queue, hessianKernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFlush(command_queue);
	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, hessian_cl_hessian, CL_TRUE, 0, sizeof(cl_double) * (numOfDampers * 7) * (numOfDampers * 7), h, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	for (register int i = 0; i < numOfDampers * 7; i++) {
		for (register int j = 0; j < numOfDampers * 7; j++) {
			hessian(i, j) = h[j + i * numOfDampers * 7];
		}
	}

	free(grad);
	free(h);
	free(det);

	return hessian;
}

void createAndBuildHessianKernel(cl_matrix2x2* sweepAlphaArr, Matrix wt, Matrix st, cl_double derH, cl_double* dampx, Matrix wtBounds, cl_double* du) {
	FILE* fp;
	size_t source_size;
	char* source_str;

	if (isConstantControl) {
		const char* filename = "constHessian.cl";
		fp = fopen(filename, "r");
	} else {
		const char* filename = "empHessian.cl";
		fp = fopen(filename, "r");
	}
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
	hessian_cl_et = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * 7, NULL, NULL);
	hessian_cl_wt_bounds = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	hessian_cl_st = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_matrix2x1) * numOfDampers, NULL, NULL);
	hessian_cl_damper_x = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers, NULL, NULL);
	hessian_cl_u = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);
	hessian_cl_v = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1), NULL, NULL);

	cl_buffer = clCreateBuffer(hessianContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);

	clSetKernelArg(hessianKernel, 0, sizeof(cl_mem), &hessian_cl_sweep_alpha);
	clSetKernelArg(hessianKernel, 1, sizeof(cl_mem), &hessian_cl_gradient);
	clSetKernelArg(hessianKernel, 2, sizeof(cl_mem), &hessian_cl_hessian);
	clSetKernelArg(hessianKernel, 3, sizeof(cl_mem), &hessian_cl_et);
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

double euclideanEmpiricalNorm(Matrix gradient, int k) {
	double n = 0.0;
	for (register int i = 0; i < numOfDampers * 7; i++) {
		n += pow(gradient(i), 2);
	}
	return sqrt(n);
}

Matrix gpuGradientAt(cl_matrix2x2* sweepAlphaArr, cl_double sei, Matrix et, Matrix st, double derH, cl_double* dampx, Matrix wtBounds, cl_double* u, cl_double* v, cl_bool lockAmplitudeParams) {
	size_t block = 7 * numOfDampers;
	//size_t block = 1;
	size_t global_work_size[] = {block, 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_double* grad = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * 7);
	cl_double* det = (cl_double*) malloc(sizeof(cl_double) * numOfDampers * 7);
	for (register int i = 0; i < numOfDampers; i++) {
		for (register int j = 0; j < 7; j++) {
			det[j + i * 7] = et(j, i);
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

	Matrix gradient(numOfDampers * 7, 1);

	clSetKernelArg(gradientKernel, 2, sizeof(cl_double), &sei);
	//clSetKernelArg(gradientKernel, 13, sizeof(cl_bool), &lockAmplitudeParams);

	cl_command_queue command_queue = clCreateCommandQueue(gradientContext, device_id, 0, &ret);

	clEnqueueWriteBuffer(command_queue, grad_cl_sweep_alpha, CL_TRUE, 0, sizeof(cl_matrix2x2) * (n - 2), sweepAlphaArr, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_wt_bounds, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmbounds, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_et, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * 7, det, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_st, CL_TRUE, 0, sizeof(cl_matrix2x1) * numOfDampers, cmst, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_damper_x, CL_TRUE, 0, sizeof(cl_double) * numOfDampers, dampx, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1), u, 0, NULL, NULL);
	clEnqueueWriteBuffer(command_queue, grad_cl_v, CL_TRUE, 0, sizeof(cl_double) * (n + 1), v, 0, NULL, NULL);

	clEnqueueNDRangeKernel(command_queue, gradientKernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

	clFlush(command_queue);
	clFinish(command_queue);

	clEnqueueReadBuffer(command_queue, grad_cl_gradient, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * 7, grad, 0, NULL, NULL);

	clReleaseCommandQueue(command_queue);

	for (register int i = 0; i < numOfDampers * 7; i++) {
		gradient(i) = grad[i];
	}

	free(grad);
	free(det);

	return gradient;
}

DEFUN_DLD(gpuEmpiricalApproach, args, nargout, "") {
	Matrix damperX = args(0).matrix_value();
	Matrix wtBounds = args(1).matrix_value();
	Matrix wt = args(2).matrix_value();
	Matrix st = args(3).matrix_value();

	numOfDampers = args(4).int_value();
	n = args(5).int_value();
	k = args(6).int_value();

	Matrix u = args(7).matrix_value();
	Matrix v = args(8).matrix_value();

	a = args(9).double_value();
	l = args(10).double_value();
	t = args(11).double_value();

	int m = args(12).int_value();
	double vareps = args(13).double_value();

	cl_bool lockAmplitudeParams = args(14).bool_value();

	cl_double* du = toDouble(u, n, k);
	cl_double* dv = toDouble(v, n, k);

	cl_double* dampx = (cl_double*) malloc(sizeof(cl_double) * numOfDampers);
	for (register int i = 0; i < numOfDampers; i++) {
		dampx[i] = damperX(i);
	}

	for (register int i = 0; i < numOfDampers; i++) {
		if (!isnan(wtBounds(i, 0)) || !isnan(wtBounds(i, 1))) {
			isPenaltyMinimization = true;
			break;
		}
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
	//cl_double* dst = (cl_double*) malloc(sizeof(cl_matrix2x1) * numOfDampers;
	for (register int i = 0; i < numOfDampers; i++) {
		for (register int j = 0; j < k + 1; j++) {
			dwt[j + i * (k + 1)] = wt(j, i);
			//dst[j + i * (k + 1)] = st(j, i);
		}
	}

	if (isPenaltyMinimization) {
		printf("\nUsing penalty method minimization\n");
	}

	printf("\nUsing Marquardt minimization\n\n");

	int iter = 0;

	cl_matrix2x2* sweepAlphaArr = solveAlpha(dampx, dwt, du);
	double fPrev = solveEmpiricalEnergyInt(sweepAlphaArr, dampx, dwt, wtBounds, st, du, dv, true);
	double fNew = fPrev;

	if (isPenaltyMinimization) {
		printf("Iteration %d/%d error with penalty: %e\n", iter, m, fPrev);
		printf("\n");

		double fPrevNoPen = solveEmpiricalEnergyInt(sweepAlphaArr, dampx, dwt, wtBounds, st, du, dv, false);
		printf("Iteration %d/%d error: %e\n", iter, m, fPrevNoPen);
		printf("\n");
	} else {
		printf("Iteration %d/%d error: %e\n", iter, m, fPrev);
		printf("\n");
	}

	double derH = pow(10, -4);

	createAndBuildGradientKernel();
	createAndBuildHessianKernel(sweepAlphaArr, wt, st, derH, dampx, wtBounds, du);

	Matrix gradient;
	double gradientNorm;

	gradient = gpuGradientAt(sweepAlphaArr, fPrev, wt, st, derH, dampx, wtBounds, du, dv, lockAmplitudeParams);
	printEmpiricalGradient(gradient, iter, m);

	gradientNorm = euclideanEmpiricalNorm(gradient, k);
	printf("Iteration %d/%d gradient norm: %e\n", iter, m, gradientNorm);
	printf("\n");
	//return octave_value(wt);

	double mu = pow(10, 4);

	while (gradientNorm > vareps && iter < m) {
		Matrix wtNew(7, numOfDampers);
		iter += 1;

		Matrix h = gpuHessianAt(sweepAlphaArr, gradient, wt, st, derH, dampx, wtBounds, du, dv, false, lockAmplitudeParams);
		printEmpiricalHessianMatrix(h, iter, m);
	
		printf("Iteration %d/%d μ: %lf\n", iter, m, mu);
		printf("\n");

		do {
			Matrix muEye = identity_matrix(numOfDampers * 7, numOfDampers * 7) * mu;
			Matrix hMuEye = h + muEye;
			Matrix hMuEyeInv = hMuEye.inverse();

			Matrix d = -1 * hMuEyeInv * gradient;
			for (register int i = 0; i < numOfDampers; i++) {
				for (register int j = 0; j < 7; j++) {
					wtNew(j, i) = wt(j, i) + d(j + i * 7);
				}
			}
			for (register int i = 0; i < numOfDampers * 7; i++) {
				dwt[i] = wtNew(i);
			}
			
			fNew = solveEmpiricalEnergyInt(sweepAlphaArr, dampx, dwt, wtBounds, st, du, dv, true);
			if (fNew < fPrev) {
				printf("μ = %lf / 2 -> %lf\n\n", mu, mu / 2.0);
				mu /= 2.0;
			} else {
				printf("μ = %lf * 2 -> %lf\n\n", mu, mu * 2.0);
				mu *= 2.0;
				if (mu > pow(10, 8)) {
					return octave_value(wt);
				}
			}
		} while (fNew >= fPrev);

		fPrev = fNew;
		if (fNew == -1.0) {
			continue;
		}
		
		printf("Iteration %d/%d w(t):\n", iter, m);
		for (register int i = 0; i < 7; i++) {
			for (register int j = 0; j < numOfDampers; j++) {
				printf("%e\t", wtNew(i + j * 7));
			}
			puts("");
		}
		printf("\n");

		if (isPenaltyMinimization) {
			printf("Iteration %d/%d error with penalty: %e\n", iter, m, fNew);
			printf("\n");

			double fNewNoPen = solveEmpiricalEnergyInt(sweepAlphaArr, dampx, dwt, wtBounds, st, du, dv, false);
			printf("Iteration %d/%d error: %e\n", iter, m, fNewNoPen);
			printf("\n");
		} else {
			printf("Iteration %d/%d error: %e\n", iter, m, fNew);
			printf("\n");
		}

		wt = wtNew;
		gradient = gpuGradientAt(sweepAlphaArr, fPrev, wt, st, derH, dampx, wtBounds, du, dv, lockAmplitudeParams);
		printEmpiricalGradient(gradient, iter, m);

		gradientNorm = euclideanEmpiricalNorm(gradient, k);
		printf("Iteration %d/%d gradient norm: %e\n", iter, m, gradientNorm);
		printf("\n");

		double hessianNorm = euclideanEmpiricalHessianNorm(h, k);
		printf("Iteration %d/%d hessian norm: %e\n", iter, m, hessianNorm);
		printf("\n");		
	}

	free(du);
	free(dv);
	free(dampx);
	free(dwt);
	free(sweepAlphaArr);
	releaseAll();
	return octave_value(wt);
}

