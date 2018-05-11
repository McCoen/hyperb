#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

typedef struct alphastruct {
	double a, b, c, d;
} cl_matrix2x2;

typedef struct betastruct {
	double a, b;
} cl_matrix2x1;

double penaltySumTwo(__global double* wt, __global cl_matrix2x1* wtBounds) {
	double g = 0.0, pen = 0.0;
	for (int i = 0; i < NUMBER_OF_DAMPERS; i++) {
		if (!isnan(wtBounds[i].a)) {
			for (int j = 0; j <= K; j++) {
				pen = wtBounds[i].a - wt[j + i * (K + 1)];
				if (pen > 0.0) {
					g += (pen * pen);
				}
			}
		}
		if (!isnan(wtBounds[i].b)) {
			for (int j = 0; j <= K; j++) {
				pen = wt[j + i * (K + 1)] - wtBounds[i].b;
				if (pen > 0.0) {
					g += (pen * pen);
				}
			}
		}
	}
	return g;
}

double penaltySumOne() {
	return 0.0;
}

double penalty(__global double* wt, __global cl_matrix2x1* wtBounds, double r, double c) {
	return r * (penaltySumOne() + penaltySumTwo(wt, wtBounds)) / 2.0;
}

cl_matrix2x2 matrixSubtract2x2(cl_matrix2x2 a, cl_matrix2x2 b) {
	a.a -= b.a;
	a.b -= b.b;
	a.c -= b.c;
	a.d -= b.d;
	return a;
}

cl_matrix2x2 matrixInverse(cl_matrix2x2 a) {
	double detDenominator = a.a * a.d - a.b * a.c;
	cl_matrix2x2 b;
	b.a = a.d / detDenominator;
	b.b = -a.b / detDenominator;
	b.c = -a.c / detDenominator;
	b.d = a.a / detDenominator;
	return b;
}

cl_matrix2x1 matrixMultiply(cl_matrix2x2 a, cl_matrix2x1 b) {
	cl_matrix2x1 c;
	c.a = a.a * b.a + a.b * b.b;
	c.b = a.c * b.a + a.d * b.b;
	return c;
}

cl_matrix2x2 matrixMultiply2x2(cl_matrix2x2 a, cl_matrix2x2 b) {
	cl_matrix2x2 c;
	c.a = a.a * b.a + a.b * b.c;
	c.b = a.a * b.b + a.b * b.d;
	c.c = a.c * b.a + a.d * b.c;
	c.d = a.c * b.b + a.d * b.d;
	return c;
}

cl_matrix2x2 matrixMultiplyByNumber2x2(cl_matrix2x2 a, double d) {
	cl_matrix2x2 b;
	b.a = a.a * d;
	b.b = a.b * d;
	b.c = a.c * d;
	b.d = a.d * d;
	return b;
}

double oscF(__global double* gu, __global double* wt, __global cl_matrix2x1* st, __global double* damperX, double a, double l, double x, const int i, const int j) {
	if (!isnan(damperX[0])) {
		double f = 0.0;

		for (int m = 0; m < NUMBER_OF_DAMPERS; m++) {
			double s = 0.0;

			double fNew = -x / (a * l) * (l - damperX[m] - s);
			if (x >= damperX[m] + s) {
				fNew += (x - damperX[m] - s) / a;
			}

			f += fNew * wt[i + m * (K + 1)];
		}

		return f;
	} else {
		return wt[i] * sign(gu[j]);
	}
}

double solveEnergyInt(__global cl_matrix2x2* alphaArr, __global double* wt, __global cl_matrix2x1* wtBounds, __global cl_matrix2x1* st, __global double* damperX, __global double* gu, __global double* gv, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 cTilde, double a, double t, double l, double hx, double ht, double alpha, double beta) {
	double f[N + 1];

	for (int i = 1; i <= K; i++) {
		cl_matrix2x1 sweepBetaArr[N - 1];
		//gv[0 + (i - 1) * (N + 1)] = 0.0;
		//gv[N + (i - 1) * (N + 1)] = 0.0;

		for (int j = 1; j < N; j++) {
			double2 y0, y1, y2;
			y0.x = gu[j - 1 + (i - 1) * (N + 1)];
			y0.y = gv[j - 1 + (i - 1) * (N + 1)];

			y1.x = gu[j + (i - 1) * (N + 1)];
			y1.y = gv[j + (i - 1) * (N + 1)];

			y2.x = gu[j + 1 + (i - 1) * (N + 1)];
			y2.y = gv[j + 1 + (i - 1) * (N + 1)];

			double2 temp1, temp2, temp3;
			temp1.x = cTilde.a * y1.x + cTilde.b * y1.y;
			temp1.y = cTilde.c * y1.x + cTilde.d * y1.y;

			temp2.x = y0.x;
			temp2.y = y0.y;

			temp2 -= temp1;
			temp2 += y2;

			double x = hx * j;
			double fA = oscF(gu, wt, st, damperX, a, l, x, i - 1, j);
			double fB = oscF(gu, wt, st, damperX, a, l, x, i, j);

			double2 currentV;
			currentV.x = -fA - fB;
			currentV.y = 0.0;

			currentV.x *= beta;
			currentV.y *= beta;

			temp2.x += currentV.x;
			temp2.y += currentV.y;

			if (j == 1) {
				sweepBetaArr[0].a = cInverted.a * temp2.x + cInverted.b * temp2.y;
				sweepBetaArr[0].b = cInverted.c * temp2.x + cInverted.d * temp2.y;
			} else {
				cl_matrix2x2 temp, invTemp;
				temp = matrixSubtract2x2(c, alphaArr[j - 2]);
				invTemp = matrixInverse(temp);

				temp3.x = sweepBetaArr[j - 2].a;
				temp3.y = sweepBetaArr[j - 2].b;
				temp3 += temp2;

				sweepBetaArr[j - 1].a = invTemp.a * temp3.x + invTemp.b * temp3.y;
				sweepBetaArr[j - 1].b = invTemp.c * temp3.x + invTemp.d * temp3.y;
			}
		}

		gu[N - 1 + i * (N + 1)] = sweepBetaArr[N - 2].a;
		gv[N - 1 + i * (N + 1)] = sweepBetaArr[N - 2].b;
		//gv[0 + i * (N + 1)] = 0.0;
		//gv[N + i * (N + 1)] = 0.0;
		for (int j = N - 2; j > 0; j--) {
			gu[j + i * (N + 1)] = alphaArr[j - 1].a * gu[j + 1 + i * (N + 1)] + alphaArr[j - 1].b * gv[j + 1 + i * (N + 1)] + sweepBetaArr[j - 1].a;
			gv[j + i * (N + 1)] = alphaArr[j - 1].c * gu[j + 1 + i * (N + 1)] + alphaArr[j - 1].d * gv[j + 1 + i * (N + 1)] + sweepBetaArr[j - 1].b;
		}
	}

	for (int i = 0; i < N + 1; i++) {
		double uTT = pow((gu[i + (K - 2) * (N + 1)] - 4.0 * gu[i + (K - 1) * (N + 1)] + 3.0 * gu[i + K * (N + 1)]) / (2.0 * ht), 2.0);
		double uXX = 0.0;
		if (i == 0) {
			uXX = (-3.0 * gu[i + K * (N + 1)] + 4.0 * gu[i + 1 + K * (N + 1)] - gu[i + 2 + K * (N + 1)]) / (2.0 * hx);
		} else if (i == N) {
			uXX = (gu[N - 2 + K * (N + 1)] - 4.0 * gu[N - 1 + K * (N + 1)] + 3.0 * gu[N + K * (N + 1)]) / (2.0 * hx);
		} else {
			uXX = (gu[i - 1 + K * (N + 1)] - 2.0 * gu[i + K * (N + 1)] + gu[i + 1 + K * (N + 1)]) / (hx * hx);
		}
		f[i] = uTT + pow(uXX, 2.0);
	}
	double numInt = f[0];
	for (int i = 1; i < N; i += 2) {
		numInt += 4.0 * f[i];
	}
	for (int i = 2; i < N; i += 2) {
		numInt += 2.0 * f[i];
	}
	numInt += f[N];
	numInt *= hx / 3.0;

	numInt += penalty(wt, wtBounds, 1.0, 5.0);

	return numInt;
}

__kernel void energyInt(__global cl_matrix2x2* sweepAlphaArr, __global double* energyInt, __global double* wt, __global cl_matrix2x1* wtBounds, __global cl_matrix2x1* st, __global double* damperX, __global double* u, __global double* v, double a, double l, double t) {

	double hx = l / N;
	double ht = t / K;

	double alpha = 2.0 * pow(hx, 2) / (a * ht);
	double beta = pow(hx, 2) / a;

	cl_matrix2x2 e2;
	e2.a = 2.0;
	e2.b = 0.0;
	e2.c = 0.0;
	e2.d = 2.0;

	cl_matrix2x2 bAlpha;
	bAlpha.a = 0.0;
	bAlpha.b = -alpha;
	bAlpha.c = alpha;
	bAlpha.d = 0.0;

	cl_matrix2x2 c;
	c.a = e2.a + bAlpha.a;
	c.b = e2.b + bAlpha.b;
	c.c = e2.c + bAlpha.c;
	c.d = e2.d + bAlpha.d;

	cl_matrix2x2 cInverted = matrixInverse(c);

	cl_matrix2x2 cTilde;
	cTilde.a = e2.a - bAlpha.a;
	cTilde.b = e2.b - bAlpha.b;
	cTilde.c = e2.c - bAlpha.c;
	cTilde.d = e2.d - bAlpha.d;

	double sei = solveEnergyInt(sweepAlphaArr, wt, wtBounds, st, damperX, u, v, c, cInverted, cTilde, a, t, l, hx, ht, alpha, beta);
	
	energyInt[0] = sei;
}

