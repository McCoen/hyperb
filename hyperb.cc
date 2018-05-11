#include <octave-4.2.1/octave/oct.h>
#include <octave-4.2.1/octave/parse.h>

#include "CL/cl.h"

cl_platform_id platform_id = NULL;
cl_device_id device_id = NULL;
cl_command_queue command_queue = NULL;
cl_mem memobj = NULL;
cl_uint ret_num_devices;
cl_uint ret_num_platforms;
cl_int ret;

int n, k, numOfDampers;
cl_double a, l, t;

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

void printGrid(double* u, int n, int k) {
	for (register int i = 0; i < k + 1; i++) {
		for (register int j = 0; j < n + 1; j++) {
			printf("%lf ", u[j + i * (n + 1)]);
		}
		puts("");
	}
}
