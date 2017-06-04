#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NUMBER_OF_DAMPERS 1
#define N 80
#define K 32

typedef struct alphastruct {
	double a, b, c, d;
} cl_matrix2x2;

typedef struct betastruct {
	double a, b;
} cl_matrix2x1;

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

cl_matrix2x2 sweepAlpha(cl_matrix2x2 sweepAlphaArr, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 sweepSide) {
	cl_matrix2x2 alpha;
	cl_matrix2x2 temp = c;
	cl_matrix2x2 temp2 = matrixMultiply2x2(sweepSide, sweepAlphaArr);
	temp.a -= temp2.a;
	temp.b -= temp2.b;
	temp.c -= temp2.c;
	temp.d -= temp2.d;

	cl_matrix2x2 invTemp = matrixInverse(temp);

	alpha = matrixMultiply2x2(invTemp, sweepSide);
	return alpha;
}

cl_matrix2x1 sweepBeta(cl_matrix2x2 sweepAlphaArr, cl_matrix2x1 sweepBetaArr, cl_matrix2x1 right, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 sweepSide) {
	cl_matrix2x1 beta;

	cl_matrix2x2 temp = c;
	cl_matrix2x2 temp2 = matrixMultiply2x2(sweepSide, sweepAlphaArr);
	temp.a -= temp2.a;
	temp.b -= temp2.b;
	temp.c -= temp2.c;
	temp.d -= temp2.d;

	cl_matrix2x2 invTemp = matrixInverse(temp);
	cl_matrix2x1 temp3 = matrixMultiply(sweepSide, sweepBetaArr);
	temp3.a += right.a;
	temp3.b += right.b;

	beta = matrixMultiply(invTemp, temp3);
	return beta;
}

double justSolveEnergyInt(__global double* wt, __global double* damperX, __global double* pu, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 cTilde, cl_matrix2x2 sweepSide, double a, double t, double l, double hx, double ht, double alpha, double beta) {

	double u[N + 1];
	double v[N + 1];
	double f[N + 1];

	for (int i = 0; i < N + 1; i++) {
		u[i] = pu[i];
		v[i] = 0.0;
		f[i] = 0.0;
	}

	for (int i = 1; i <= K; i++) {
		cl_matrix2x2 sweepAlphaArr[N - 2];
		for (int j = 0; j < N - 2; j++) {
			if (j == 0) {
				sweepAlphaArr[0] = matrixMultiply2x2(cInverted, sweepSide);
			} else {
				sweepAlphaArr[j] = sweepAlpha(sweepAlphaArr[j - 1], c, cInverted, sweepSide);
			}
		}

		cl_matrix2x1 sweepBetaArr[N - 1];
		for (int j = 1; j < N; j++) {
			cl_matrix2x1 y0, y1, y2;
			y0.a = u[j - 1];
			y0.b = v[j - 1];
			y1.a = u[j];
			y1.b = v[j];
			y2.a = u[j + 1];
			y2.b = v[j + 1];

			cl_matrix2x1 temp1 = matrixMultiply(cTilde, y1);

			cl_matrix2x1 temp = y0;
			temp.a -= temp1.a;
			temp.b -= temp1.b;

			temp.a += y2.a;
			temp.b += y2.b;

			cl_matrix2x1 currentV;

			double fA = 0.0, fB = 0.0;
			double x = hx * j;

			for (int m = 0; m < NUMBER_OF_DAMPERS; m++) {
				double wA = wt[(i - 1) + m * (K + 1)];
				double wB = wt[i + m * (K + 1)];
				double fNew = -x / (a * l) * (l - damperX[m]);

				if (x >= damperX[m]) {
					fNew += 1.0 / a * (x - damperX[m]);
				}

				fA += fNew * wA;
				fB += fNew * wB;
			}

			currentV.a = -fA - fB;
			currentV.b = 0.0;

			currentV.a *= beta;
			currentV.b *= beta;

			temp.a += currentV.a;
			temp.b += currentV.b;

			cl_matrix2x1 currentRight = temp;

			if (j == 1) {
				sweepBetaArr[0] = matrixMultiply(cInverted, currentRight);
			} else {
				sweepBetaArr[j - 1] = sweepBeta(sweepAlphaArr[j - 2], sweepBetaArr[j - 2], currentRight, c, cInverted, sweepSide);
			}
		}

		cl_matrix2x1 x[N - 1];
		x[N - 2] = sweepBetaArr[N - 2];

		for (int j = N - 3; j >= 0; j--) {
			cl_matrix2x1 temp = matrixMultiply(sweepAlphaArr[j], x[j + 1]);
			temp.a += sweepBetaArr[j].a;
			temp.b += sweepBetaArr[j].b;
			x[j] = temp;
		}

		for (int j = 1; j < N; j++) {
			cl_matrix2x1 current = x[j - 1];

			u[j] = current.a;
			v[j] = current.b;

		}

		if (i == K - 2) {
			for (int j = 0; j <= N; j++) {
				f[j] = u[j];
			}
		}
		if (i == K - 1) {
			for (int j = 0; j <= N; j++) {
				f[j] = f[j] - 4.0 * u[j];
			}
		}
		if (i == K) {
			for (int j = 0; j <= N; j++) {
				f[j] = f[j] + 3.0 * u[j];
				f[j] = f[j] / 2.0;
				f[j] = f[j] / ht;
				f[j] = f[j] * f[j];

				double dx = 0.0;
				if (j == 0) {
					dx = (-3.0 * u[j] + 4.0 * u[j + 1] - u[j + 2]) / (2.0 * hx);
				} else if (j == N) {
					dx = (u[N - 2] - 4.0 * u[N - 1] + 3.0 * u[N]) / (2.0 * hx);
				} else {
					dx = (u[j - 1] - 2.0 * u[j] + u[j + 1]) / (hx * hx);
				}
				dx = dx * dx;
				f[j] = f[j] + dx;
			}
		}
	}

	double numInt = f[0];

	for (int i = 1; i < N; i += 2) {
		numInt += 4 * f[i];
	}
	for (int i = 2; i < N; i += 2) {
		numInt += 2 * f[i];
	}
	numInt += f[N];

	numInt *= hx / 3.0;

	return numInt;
}

double justSolve(__global double* wt, __global double* damperX, __global double* pu, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 cTilde, cl_matrix2x2 sweepSide, double a, double t, double l, double hx, double ht, double alpha, double beta) {

	double u[N + 1];
	double v[N + 1];
	for (int i = 0; i < N + 1; i++) {
		u[i] = pu[i];
		v[i] = 0.0;
	}

	int gi = 1, gj = 0;
	for (int i = 1; i <= K; i++) {
		cl_matrix2x2 sweepAlphaArr[N - 2];
		for (int j = 0; j < N - 2; j++) {
			if (j == 0) {
				sweepAlphaArr[0] = matrixMultiply2x2(cInverted, sweepSide);
			} else {
				sweepAlphaArr[j] = sweepAlpha(sweepAlphaArr[j - 1], c, cInverted, sweepSide);
			}
		}

		cl_matrix2x1 sweepBetaArr[N - 1];
		for (int j = 1; j < N; j++) {
			cl_matrix2x1 y0, y1, y2;
			y0.a = u[j - 1];
			y0.b = v[j - 1];
			y1.a = u[j];
			y1.b = v[j];
			y2.a = u[j + 1];
			y2.b = v[j + 1];

			cl_matrix2x1 temp1 = matrixMultiply(cTilde, y1);

			cl_matrix2x1 temp = y0;
			temp.a -= temp1.a;
			temp.b -= temp1.b;

			temp.a += y2.a;
			temp.b += y2.b;

			cl_matrix2x1 currentV;

			double fA = 0.0, fB = 0.0;
			double x = hx * j;

			for (int m = 0; m < NUMBER_OF_DAMPERS; m++) {
				double wA = wt[(i - 1) + m * (K + 1)];
				double wB = wt[i + m * (K + 1)];
				double fNew = -x / (a * l) * (l - damperX[m]);

				if (x >= damperX[m]) {
					fNew += 1.0 / a * (x - damperX[m]);
				}

				fA += fNew * wA;
				fB += fNew * wB;
			}

			currentV.a = -fA - fB;
			currentV.b = 0.0;

			currentV.a *= beta;
			currentV.b *= beta;

			temp.a += currentV.a;
			temp.b += currentV.b;

			cl_matrix2x1 currentRight = temp;

			if (j == 1) {
				sweepBetaArr[0] = matrixMultiply(cInverted, currentRight);
			} else {
				sweepBetaArr[j - 1] = sweepBeta(sweepAlphaArr[j - 2], sweepBetaArr[j - 2], currentRight, c, cInverted, sweepSide);
			}
		}

		cl_matrix2x1 x[N - 1];
		x[N - 2] = sweepBetaArr[N - 2];

		for (int j = N - 3; j >= 0; j--) {
			cl_matrix2x1 temp = matrixMultiply(sweepAlphaArr[j], x[j + 1]);
			temp.a += sweepBetaArr[j].a;
			temp.b += sweepBetaArr[j].b;
			x[j] = temp;
		}

		for (int j = 1; j < N; j++) {
			cl_matrix2x1 current = x[j - 1];

			u[j] = current.a;
			v[j] = current.b;

		}

		gj = 0;
		for (int j = 0; j <= N; j++) {
			pu[gj + gi * (N + 1)] = u[gj];
			gj++;
		}
		gi++;
	}

	return 0.0;
}

__kernel void gradientAt(__global double* grad, __global double* energyInt, __global double* wt, __global double* damperX, __global double* u, __global double* v, __global cl_matrix2x1* sweepRightArr, __global cl_matrix2x2* sweepAlphaArr, __global cl_matrix2x1* sweepBetaArr, double a, double l, double t, __global int* noFault) {

	//TODO change here
	a = 1.0;
	l = 1.0;

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

	cl_matrix2x2 sweepSide;
	sweepSide.a = 1.0;
	sweepSide.b = 0.0;
	sweepSide.c = 0.0;
	sweepSide.d = 1.0;

	double sei = justSolveEnergyInt(wt, damperX, u, c, cInverted, cTilde, sweepSide, a, t, l, hx, ht, alpha, beta);

	energyInt[0] = sei;

	noFault[0] = 1;
}

__kernel void energyInt(__global double* energyInt, __global double* wt, __global double* damperX, __global double* u, __global double* v, __global cl_matrix2x1* sweepRightArr, __global cl_matrix2x2* sweepAlphaArr, __global cl_matrix2x1* sweepBetaArr, double a, double l, double t, __global int* noFault) {

	//TODO change here
	a = 1.0;
	l = 1.0;

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

	cl_matrix2x2 sweepSide;
	sweepSide.a = 1.0;
	sweepSide.b = 0.0;
	sweepSide.c = 0.0;
	sweepSide.d = 1.0;

	double sei = justSolveEnergyInt(wt, damperX, u, c, cInverted, cTilde, sweepSide, a, t, l, hx, ht, alpha, beta);
	
	energyInt[0] = sei;

	noFault[0] = 1;
}

__kernel void solveGrid(__global double* energyInt, __global double* wt, __global double* damperX, __global double* u, __global double* v, __global cl_matrix2x1* sweepRightArr, __global cl_matrix2x2* sweepAlphaArr, __global cl_matrix2x1* sweepBetaArr, double a, double l, double t, __global int* noFault) {

	//TODO change here
	a = 1.0;
	l = 1.0;

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

	cl_matrix2x2 sweepSide;
	sweepSide.a = 1.0;
	sweepSide.b = 0.0;
	sweepSide.c = 0.0;
	sweepSide.d = 1.0;

	double sei = justSolve(wt, damperX, u, c, cInverted, cTilde, sweepSide, a, t, l, hx, ht, alpha, beta);

	energyInt[0] = sei;

	noFault[0] = 1;
}
