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
	releaseGradientKernel();
	releaseHessianKernel();
}

DEFUN_DLD(gpuMarquardtMinimization, args, nargout, "") {
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

	cl_double* du = toDouble(u, n, k);
	cl_double* dv = toDouble(v, n, k);

	cl_double* dampx = (cl_double*) malloc(sizeof(cl_double) * numOfDampers);
	for (register int i = 0; i < numOfDampers; i++) {
		dampx[i] = damperX(i);
	}

	bool isPenaltyMinimization = false;
	for (register int i = 0; i < numOfDampers; i++) {
		if (!isnan(wtBounds(i, 0)) || !isnan(wtBounds(i, 1))) {
			isPenaltyMinimization = true;
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

	cl_matrix2x2* sweepAlphaArr = solveAlpha(dampx, dwt);
	double fPrev = solveEnergyInt(sweepAlphaArr, dampx, dwt, wtBounds, st, du, dv, true);
	double fNew = fPrev;

	if (isPenaltyMinimization) {
		printf("Iteration %d/%d error with penalty: %e\n", iter, m, fPrev);
		printf("\n");

		double fPrevNoPen = solveEnergyInt(sweepAlphaArr, dampx, dwt, wtBounds, st, du, dv, false);
		printf("Iteration %d/%d error: %e\n", iter, m, fPrevNoPen);
		printf("\n");
	} else {
		printf("Iteration %d/%d error: %e\n", iter, m, fPrev);
		printf("\n");
	}

	double derH = pow(10, -4);

	createAndBuildGradientKernel();
	createAndBuildHessianKernel(sweepAlphaArr, wt, st, derH, dampx, wtBounds, du);
	//createAndBuildNewHessianKernel();

	Matrix gradient;
	double gradientNorm;

	gradient = gpuGradientAt(sweepAlphaArr, fPrev, wt, st, derH, dampx, wtBounds, du, dv);
	//printGradient(gradient, iter, m);

	gradientNorm = euclideanGradientNorm(gradient, k);
	printf("Iteration %d/%d gradient norm: %e\n", iter, m, gradientNorm);
	printf("\n");

	double mu = pow(10, 4);

	while (gradientNorm > vareps && iter < m) {
		Matrix wtNew(k + 1, numOfDampers);
		iter += 1;

		Matrix h = gpuHessianAt(sweepAlphaArr, gradient, wt, st, derH, dampx, wtBounds, du, dv, false);
		//printHessianMatrix(h, iter, m);

		printf("Iteration %d/%d μ: %lf\n", iter, m, mu);
		printf("\n");

		do {
			Matrix muEye = identity_matrix(numOfDampers * (k + 1), numOfDampers * (k + 1)) * mu;
			Matrix hMuEye = h + muEye;
			Matrix hMuEyeInv = hMuEye.inverse();

			Matrix d = -1 * hMuEyeInv * gradient;
			for (register int i = 0; i < numOfDampers; i++) {
				for (register int j = 0; j < k + 1; j++) {
					wtNew(j, i) = wt(j, i) + d(j + i * (k + 1));
				}
			}
			for (register int i = 0; i < numOfDampers * (k + 1); i++) {
				dwt[i] = wtNew(i);
			}
			
			fNew = solveEnergyInt(sweepAlphaArr, dampx, dwt, wtBounds, st, du, dv, true);
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
		for (register int i = 0; i < k + 1; i++) {
			for (register int j = 0; j < numOfDampers; j++) {
				printf("%e\t", wtNew(i + j * (k + 1)));
			}
			puts("");
		}
		printf("\n");

		if (isPenaltyMinimization) {
			printf("Iteration %d/%d error with penalty: %e\n", iter, m, fNew);
			printf("\n");

			double fNewNoPen = solveEnergyInt(sweepAlphaArr, dampx, dwt, wtBounds, st, du, dv, false);
			printf("Iteration %d/%d error: %e\n", iter, m, fNewNoPen);
			printf("\n");
		} else {
			printf("Iteration %d/%d error: %e\n", iter, m, fNew);
			printf("\n");
		}

		wt = wtNew;
		gradient = gpuGradientAt(sweepAlphaArr, fPrev, wt, st, derH, dampx, wtBounds, du, dv);
		//printGradient(gradient, iter, m);

		gradientNorm = euclideanGradientNorm(gradient, k);
		printf("Iteration %d/%d gradient norm: %e\n", iter, m, gradientNorm);
		printf("\n");

		double hessianNorm = euclideanHessianNorm(h, k);
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
