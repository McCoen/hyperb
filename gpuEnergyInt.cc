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

using namespace std;

void releaseAll() {
	releaseSweepAlphaKernel();
	releaseKernel();
}

DEFUN_DLD(gpuEnergyInt, args, nargout, "") {
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
	double err = solveEnergyInt(sweepAlphaArr, dampx, dwt, Matrix(1, 1), st, du, dv, false);

	free(du);
	free(dv);
	free(dampx);
	free(dwt);
	free(dst);
	free(sweepAlphaArr);
	releaseAll();

	return octave_value(err);
}

