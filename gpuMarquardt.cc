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
cl_context context = NULL;
cl_command_queue command_queue = NULL;
cl_mem memobj = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;

cl_context gradientContext = NULL;
cl_program gradientProgram = NULL;
cl_kernel gradientKernel = NULL;

void createAndBuildGradientKernel() {
	FILE* fp;
	const char* filename = "gradient.cl";
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

	gradientContext = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	gradientProgram = clCreateProgramWithSource(gradientContext, 1, (const char**) &source_str, (const size_t*) &source_size, &ret);

	puts("Now building gradient kernel");

	ret = clBuildProgram(gradientProgram, 1, &device_id, NULL, NULL, NULL);
	gradientKernel = clCreateKernel(gradientProgram, "gradientAt", &ret);

	puts("OpenCL kernel has built");
}

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

	puts("Now building kernel");

	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	kernel = clCreateKernel(program, "energyInt", &ret);

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

cl_double solveEnergyInt(cl_double* damperX, cl_double* wt, cl_double* u, int n, int k) {
	size_t block = 1;
	size_t global_work_size[] = {block, 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_int* no_fault = (cl_int*) malloc(sizeof(cl_int));
	no_fault[0] = 0;

	cl_double a = 1.0;
	cl_double l = 1.0;
	cl_double t = 1.0;

	int numOfDampers = 1;

	cl_mem cl_energy_int = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double), NULL, NULL);
	cl_mem cl_damper_x = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers, NULL, NULL);
	cl_mem cl_wt = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);
	cl_mem cl_u = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);
	cl_mem cl_no_fault = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, NULL);

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_energy_int);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_wt);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_damper_x);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_u);
	clSetKernelArg(kernel, 4, sizeof(cl_double), &a);
	clSetKernelArg(kernel, 5, sizeof(cl_double), &l);
	clSetKernelArg(kernel, 6, sizeof(cl_double), &t);
	clSetKernelArg(kernel, 7, sizeof(cl_mem), &cl_no_fault);

	cl_double* sei = (cl_double*) malloc(sizeof(cl_double));
	sei[0] = -1.0;

	do {
		command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

		clEnqueueWriteBuffer(command_queue, cl_energy_int, CL_TRUE, 0, sizeof(cl_double), sei, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_wt, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * (k + 1), wt, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_damper_x, CL_TRUE, 0, sizeof(cl_double) * numOfDampers, damperX, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1), u, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_no_fault, CL_TRUE, 0, sizeof(cl_int), no_fault, 0, NULL, NULL);

		clEnqueueNDRangeKernel(command_queue, kernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

		clFlush(command_queue);
		clFinish(command_queue);

		clEnqueueReadBuffer(command_queue, cl_energy_int, CL_TRUE, 0, sizeof(cl_double), sei, 0, NULL, NULL);
		clEnqueueReadBuffer(command_queue, cl_no_fault, CL_TRUE, 0, sizeof(cl_int), no_fault, 0, NULL, NULL);

		clReleaseCommandQueue(command_queue);
	} while (no_fault[0] != 1);

	clReleaseMemObject(cl_energy_int);
	clReleaseMemObject(cl_damper_x);
	clReleaseMemObject(cl_wt);
	clReleaseMemObject(cl_u);
	clReleaseMemObject(cl_no_fault);

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

double derivativeAt(double* dampx, Matrix wt, double* du, int n, int k, int nth) {
	double derH = pow(10, -4);

	Matrix wt1 = wt;

	wt1(nth) = wt1(nth) + derH;

	double dwt[2 * (k + 1)];
	double dwt1[2 * (k + 1)];
	for (register int i = 0; i < 2 * (k + 1); i++) {
		dwt[i] = wt(i);
		dwt1[i] = wt1(i);
	}

	double f0 = solveEnergyInt(dampx, dwt, du, n, k);
	double f1 = solveEnergyInt(dampx, dwt1, du, n, k);

	return (f1 - f0) / derH;
}

Matrix gpuHessianAt(Matrix gradient, Matrix wt, int k, double derH, double* dampx, double* du, int n) {
	Matrix h(k + 1, k + 1);
	for (register int i = 0; i < k + 1; i++) {
		double d0 = gradient(i);
		for (register int j = i; j < k + 1; j++) {
			Matrix wtNew = wt;
			wtNew(j) += derH;

			double d1 = derivativeAt(dampx, wtNew, du, n, k, i);
			double sd = (d1 - d0) / derH;

			h(i, j) = sd;
			h(j, i) = sd;
		}
	}
	return h;
}

double euclideanNorm(Matrix gradient, int k) {
	double n = 0.0;
	for (register int i = 0; i < k + 1; i++) {
		n += pow(gradient(i), 2);
	}
	return sqrt(n);
}

Matrix gpuGradientAt(cl_double sei, Matrix wt, int k, double derH, cl_double* dampx, cl_double* u, int n) {
	size_t block = 1;
	size_t global_work_size[] = {block, 0, 0};
	size_t local_work_size[] = {block, 0, 0};

	cl_int* no_fault = (cl_int*) malloc(sizeof(cl_int));
	no_fault[0] = 0;

	cl_double a = 1.0;
	cl_double l = 1.0;
	cl_double t = 1.0;

	int numOfDampers = 1;

	cl_mem cl_gradient = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (k + 1), NULL, NULL);
	cl_mem cl_wt = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers * (k + 1), NULL, NULL);
	cl_mem cl_damper_x = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * numOfDampers, NULL, NULL);
	cl_mem cl_u = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_double) * (n + 1) * (k + 1), NULL, NULL);
	cl_mem cl_no_fault = clCreateBuffer(gradientContext, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, NULL);

	clSetKernelArg(gradientKernel, 0, sizeof(cl_double), &sei);
	clSetKernelArg(gradientKernel, 1, sizeof(cl_mem), &cl_gradient);
	clSetKernelArg(gradientKernel, 2, sizeof(cl_mem), &cl_wt);
	clSetKernelArg(gradientKernel, 3, sizeof(cl_mem), &cl_damper_x);
	clSetKernelArg(gradientKernel, 4, sizeof(cl_mem), &cl_u);
	clSetKernelArg(gradientKernel, 5, sizeof(cl_double), &a);
	clSetKernelArg(gradientKernel, 6, sizeof(cl_double), &l);
	clSetKernelArg(gradientKernel, 7, sizeof(cl_double), &t);
	clSetKernelArg(gradientKernel, 8, sizeof(cl_mem), &cl_no_fault);

	cl_double* grad = (cl_double*) malloc(sizeof(cl_double) * (k + 1));
	cl_double* dwt = (cl_double*) malloc(sizeof(cl_double) * (k + 1));
	for (register int i = 0; i < k + 1; i++) {
		dwt[i] = wt(i);
	}

	do {
		cl_command_queue command_queue = clCreateCommandQueue(gradientContext, device_id, 0, &ret);

		//clEnqueueWriteBuffer(command_queue, cl_gradient, CL_TRUE, 0, sizeof(cl_double) * (k + 1), gradient, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_wt, CL_TRUE, 0, sizeof(cl_double) * numOfDampers * (k + 1), dwt, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_damper_x, CL_TRUE, 0, sizeof(cl_double) * numOfDampers, dampx, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_u, CL_TRUE, 0, sizeof(cl_double) * (n + 1), u, 0, NULL, NULL);
		clEnqueueWriteBuffer(command_queue, cl_no_fault, CL_TRUE, 0, sizeof(cl_int), no_fault, 0, NULL, NULL);

		clEnqueueNDRangeKernel(command_queue, gradientKernel, CL_TRUE, NULL, global_work_size, local_work_size, 0, NULL, NULL);

		clFlush(command_queue);
		clFinish(command_queue);

		clEnqueueReadBuffer(command_queue, cl_gradient, CL_TRUE, 0, sizeof(cl_double) * (k + 1), grad, 0, NULL, NULL);
		clEnqueueReadBuffer(command_queue, cl_no_fault, CL_TRUE, 0, sizeof(cl_int), no_fault, 0, NULL, NULL);

		clReleaseCommandQueue(command_queue);
	} while (no_fault[0] != 1);

	clReleaseMemObject(cl_gradient);
	clReleaseMemObject(cl_wt);
	clReleaseMemObject(cl_damper_x);
	clReleaseMemObject(cl_u);
	clReleaseMemObject(cl_no_fault);

	Matrix gradient(k + 1, 1);
	for (register int i = 0; i < k + 1; i++) {
		gradient(i) = grad[i];
	}
	return gradient;
}

DEFUN_DLD(gpuMarquardt, args, nargout, "") {
	Matrix damperX = args(0).matrix_value();
	Matrix wt = args(1).matrix_value();
	int n = args(2).int_value();
	int k = args(3).int_value();
	Matrix u = args(4).matrix_value();
	cl_double a = args(5).double_value();
	cl_double l = args(6).double_value();
	cl_double t = args(7).double_value();

	createAndBuildKernel();
	createAndBuildGradientKernel();
	
	double* du = toDouble(u, n, k);
	double dampx[]{damperX(0), damperX(1)};
	double dwt[k + 1];
	for (register int i = 0; i < k + 1; i++) {
		dwt[i] = wt(i);
	}

	printf("\nUsing Marquardt minimization\n\n");

	int iter = 0, m = 1;

	double fPrev = solveEnergyInt(dampx, dwt, du, n, k);
	double fNew = fPrev;

	printf("Iteration %d/%d error: %lf\n", iter, m, fPrev);
	printf("\n");

	double derH = pow(10, -4);
	Matrix gradient;
	double gradientNorm;

	gradient = gpuGradientAt(fPrev, wt, k, derH, dampx, du, n);
	
	printf("Iteration %d/%d gradient:\n", iter, m);
	for (register int i = 0; i < k + 1; i++) {
		printf("%lf\n", gradient(i));
	}
	printf("\n");

	gradientNorm = euclideanNorm(gradient, k);
	printf("Iteration %d/%d gradient norm: %lf\n", iter, m, gradientNorm);
	printf("\n");

	double mu = pow(10, 4);
	Matrix wtNew;
	while (gradientNorm > pow(10, -6) && iter < m) {
		iter += 1;

		Matrix h = gpuHessianAt(gradient, wt, k, derH, dampx, du, n);
		printf("Iteration %d/%d Hessian matrix:\n", iter, m);
		for (register int i = 0; i < k + 1; i++) {
			for (register int j = 0; j < k + 1; j++) {
				printf("%lf ", h(i, j));
			}
			puts("");
		}
		printf("\n");

		printf("Iteration %d/%d μ: %lf\n", iter, m, mu);
		printf("\n");

		do {
			Matrix muEye = identity_matrix(k + 1, k + 1) * mu;
			Matrix hMuEye = h + muEye;
			Matrix hMuEyeInv = hMuEye.inverse();

			Matrix d = -1 * hMuEyeInv * gradient;
			wtNew = wt + d;
			for (register int i = 0; i < k + 1; i++) {
				dwt[i] = wtNew(i);
			}
			fNew = solveEnergyInt(dampx, dwt, du, n, k);

			
			if (fNew < fPrev) {
				printf("μ = %lf / 2 -> %lf\n\n", mu, mu / 2.0);
				mu /= 2.0;
			} else {
				printf("μ = %lf * 2 -> %lf\n\n", mu, mu * 2.0);
				mu *= 2.0;
			}
		} while (fNew >= fPrev);

		fPrev = fNew;
		
		printf("Iteration %d/%d w(t):\n", iter, m);
		for (register int i = 0; i < k + 1; i++) {
			printf("%lf\n", wtNew(i));
		}
		printf("\n");

		printf("Iteration %d/%d error: %lf\n", iter, m, fNew);
		printf("\n");

		wt = wtNew;
		gradient = gpuGradientAt(fPrev, wt, k, derH, dampx, du, n);
		printf("Iteration %d/%d gradient:\n", iter, m);
		for (register int i = 0; i < k + 1; i++) {
			printf("%lf\n", gradient(i));
		}
		printf("\n");

		gradientNorm = euclideanNorm(gradient, k);
		printf("Iteration %d/%d gradient norm: %lf\n", iter, m, gradientNorm);
		printf("\n");
	}

	/*

		output_file = fopen('marquardt_last', 'w');
		fprintf(output_file, "%.256e\n", mu);
		for i = 1 : length(wt(:))
			fprintf(output_file, "%.256e\n", wt_new(i));
		endfor
		fclose(output_file);
	
		
	*/

	releaseAll();
	return octave_value(wt);
}

