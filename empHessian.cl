#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct alphastruct {
	double a, b, c, d;
} cl_matrix2x2;

typedef struct betastruct {
	double a, b;
} cl_matrix2x1;

double getHessianWtAt(__global double* et, double ht, const int i, const int m, const int nthCol, const int nthRow, bool lockAmplitudeParams) {
	double t = i * ht;
	double currentEt[NUMBER_OF_DAMPERS * 7];

	for (int k = 0; k < NUMBER_OF_DAMPERS * 7; k++) {
		currentEt[k] = et[k];
	}
	currentEt[nthCol] += 0.0001;
	currentEt[nthRow] += 0.0001;

	if (lockAmplitudeParams) {
		currentEt[m * 7] = et[m * 7];
		currentEt[3 + m * 7] = et[3 + m * 7];
	}

	return currentEt[m * 7] * sin(currentEt[1 + m * 7] * t + currentEt[2 + m * 7]) + currentEt[3 + m * 7] * sin(currentEt[4 + m * 7] * t + currentEt[5 + m * 7]) * sin(currentEt[6 + m * 7] * t);
}

double empiricalHessianOscF(const int nthCol, const int nthRow, __global double* et, __global cl_matrix2x1* st, __global double* damperX, double a, double l, double x, double ht, const int i, bool lockAmplitudeParams) {
	double f = 0.0;

	for (int m = 0; m < NUMBER_OF_DAMPERS; m++) {
		double s = 0.0;

		double fNew = -x / (a * l) * (l - damperX[m] - s);
		if (x >= damperX[m] + s) {
			fNew += (x - damperX[m] - s) / a;
		}

		double wt = getHessianWtAt(et, ht, i, m, nthCol, nthRow, lockAmplitudeParams);
		f += fNew * wt;
	}

	return f;
}

double solveEnergyInt(const int nthCol, const int nthRow, __global cl_matrix2x2* alphaArr, __global double* et, __global cl_matrix2x1* wtBounds, __global cl_matrix2x1* st, __global double* damperX, __global double* gu, __global double* gv, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 cTilde, double a, double t, double l, double hx, double ht, double alpha, double beta, bool lockAmplitudeParams) {
	double u[N + 1];
	double v[N + 1];
	double f[N + 1];
	for (int i = 0; i < N + 1; i++) {
		u[i] = gu[i];
		v[i] = gv[i];
	}

	for (int i = 1; i <= K; i++) {
		cl_matrix2x1 sweepBetaArr[N - 1];

		for (int j = 1; j < N; j++) {
			

			if (j == 1) {
				double2 y0, y1, y2;

				y0.x = u[j - 1];
				y0.y = v[j - 1];

				y1.x = u[j];
				y1.y = v[j];

				y2.x = u[j + 1];
				y2.y = v[j + 1];

				double2 temp1, temp2, temp3;

				temp1.x = cTilde.a * y1.x + cTilde.b * y1.y;
				temp1.y = cTilde.c * y1.x + cTilde.d * y1.y;

				temp2.x = y0.x;
				temp2.y = y0.y;

				temp2.x -= temp1.x;
				temp2.y -= temp1.y;

				temp2.x += y2.x;
				temp2.y += y2.y;

				double x = hx * j;
				double fA = empiricalHessianOscF(nthCol, nthRow, et, st, damperX, a, l, x, ht, i - 1, lockAmplitudeParams);
				double fB = empiricalHessianOscF(nthCol, nthRow, et, st, damperX, a, l, x, ht, i, lockAmplitudeParams);

				double2 currentV;
				currentV.x = -fA - fB;
				currentV.y = 0.0;

				currentV.x *= beta;
				currentV.y *= beta;

				temp2.x += currentV.x;
				temp2.y += currentV.y;

				sweepBetaArr[0].a = cInverted.a * temp2.x + cInverted.b * temp2.y;
				sweepBetaArr[0].b = cInverted.c * temp2.x + cInverted.d * temp2.y;
			} else {
				double2 y0, y1, y2;

				y0.x = u[j - 1];
				y0.y = v[j - 1];

				y1.x = u[j];
				y1.y = v[j];

				y2.x = u[j + 1];
				y2.y = v[j + 1];

				double2 temp1, temp2, temp3;

				temp1.x = cTilde.a * y1.x + cTilde.b * y1.y;
				temp1.y = cTilde.c * y1.x + cTilde.d * y1.y;

				temp2.x = y0.x;
				temp2.y = y0.y;

				temp2.x -= temp1.x;
				temp2.y -= temp1.y;

				temp2.x += y2.x;
				temp2.y += y2.y;

				double x = hx * j;
				double fA = empiricalHessianOscF(nthCol, nthRow, et, st, damperX, a, l, x, ht, i - 1, lockAmplitudeParams);
				double fB = empiricalHessianOscF(nthCol, nthRow, et, st, damperX, a, l, x, ht, i, lockAmplitudeParams);

				double2 currentV;
				currentV.x = -fA - fB;
				currentV.y = 0.0;

				currentV.x *= beta;
				currentV.y *= beta;

				temp2.x += currentV.x;
				temp2.y += currentV.y;

				cl_matrix2x2 temp, invTemp;
				temp.a = c.a - alphaArr[j - 2].a;
				temp.b = c.b - alphaArr[j - 2].b;
				temp.c = c.c - alphaArr[j - 2].c;
				temp.d = c.d - alphaArr[j - 2].d;

				double detDenominator = temp.a * temp.d - temp.b * temp.c;
				invTemp.a = temp.d / detDenominator;
				invTemp.b = -temp.b / detDenominator;
				invTemp.c = -temp.c / detDenominator;
				invTemp.d = temp.a / detDenominator;

				temp3.x = sweepBetaArr[j - 2].a;
				temp3.y = sweepBetaArr[j - 2].b;
				temp3.x += temp2.x;
				temp3.y += temp2.y;

				sweepBetaArr[j - 1].a = invTemp.a * temp3.x + invTemp.b * temp3.y;
				sweepBetaArr[j - 1].b = invTemp.c * temp3.x + invTemp.d * temp3.y;
			}
		}

		u[N - 1] = sweepBetaArr[N - 2].a;
		v[N - 1] = sweepBetaArr[N - 2].b;
		v[0] = 0.0;
		v[N] = 0.0;

		for (int j = N - 2; j > 0; j--) {
			u[j] = alphaArr[j - 1].a * u[j + 1] + alphaArr[j - 1].b * v[j + 1] + sweepBetaArr[j - 1].a;
			v[j] = alphaArr[j - 1].c * u[j + 1] + alphaArr[j - 1].d * v[j + 1] + sweepBetaArr[j - 1].b;
		}

		if (i == K - 2) {
			for (int j = 0; j <= N; j++) {
				f[j] = u[j];
			}
		} else if (i == K - 1) {
			for (int j = 0; j <= N; j++) {
				f[j] = f[j] - 4.0 * u[j];
			}
		} else if (i == K) {
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
		numInt += 4.0 * f[i];
	}
	for (int i = 2; i < N; i += 2) {
		numInt += 2.0 * f[i];
	}
	numInt += f[N];
	numInt *= hx / 3.0;

	//numInt += hessianPenalty(wt, wtBounds, 1.0, 5.0, nthCol, nthRow);

	return numInt;
}

__kernel void hessianAt(__global cl_matrix2x2* sweepAlphaArr, __global double* grad, __global double* h, __global double* et, __global cl_matrix2x1* wtBounds, __global cl_matrix2x1* st, __global double* damperX, __global double* u, __global double* v, double a, double l, double t, __global double* buffer) {
	double derH = 0.0001;
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

	cl_matrix2x2 cInverted;
	double detDenominator = c.a * c.d - c.b * c.c;
	cInverted.a = c.d / detDenominator;
	cInverted.b = -c.b / detDenominator;
	cInverted.c = -c.c / detDenominator;
	cInverted.d = c.a / detDenominator;

	cl_matrix2x2 cTilde;
	cTilde.a = e2.a - bAlpha.a;
	cTilde.b = e2.b - bAlpha.b;
	cTilde.c = e2.c - bAlpha.c;
	cTilde.d = e2.d - bAlpha.d;

	int nthCol = get_global_id(0) % (NUMBER_OF_DAMPERS * 7);
	int nthRow = get_global_id(0) / (NUMBER_OF_DAMPERS * 7);

	bool lockAmplitudeParams = true;
	if (nthCol >= nthRow) {
		double hess1 = solveEnergyInt(nthCol, -1, sweepAlphaArr, et, wtBounds, st, damperX, u, v, c, cInverted, cTilde, a, t, l, hx, ht, alpha, beta, lockAmplitudeParams);
		double hess2 = solveEnergyInt(nthCol, nthRow, sweepAlphaArr, et, wtBounds, st, damperX, u, v, c, cInverted, cTilde, a, t, l, hx, ht, alpha, beta, lockAmplitudeParams);
		
		h[nthCol + nthRow * NUMBER_OF_DAMPERS * 7] = ((hess2 - hess1) / 0.0001 - grad[nthRow]) / 0.0001;
		h[nthRow + nthCol * NUMBER_OF_DAMPERS * 7] = h[nthCol + nthRow * NUMBER_OF_DAMPERS * 7];
	}
}
