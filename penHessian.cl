#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef struct alphastruct {
	double a, b, c, d;
} cl_matrix2x2;

typedef struct betastruct {
	double a, b;
} cl_matrix2x1;

double penaltySumTwo(double* wt, double* damperX, double* wtBounds) {
	double g = 0.0, pen = 0.0;
	for (int i = 0; i < NUMBER_OF_DAMPERS; i++) {
		for (int j = 0; j <= K; j++) {
			pen = wtBounds[i * 2] - wt[j + i * (K + 1)];
			if (pen > 0.0) {
				g += (pen * pen);
			}
		}
		for (int j = 0; j <= K; j++) {
			pen = wt[j + i * (K + 1)] - wtBounds[1 + i * 2];
			if (pen > 0.0) {
				g += (pen * pen);
			}
		}
	}
	return g;
}

double penaltySumOne() {
	return 0.0;
}

double penalty(double* wt, double* st, double* damperX, double* wtBounds, double r, double c) {
	return r * (penaltySumOne() + penaltySumTwo(wt, damperX, wtBounds)) / 2.0;
}

double oscF(double* wt, double* st, double* damperX, double a, double l, double x, const int i) {
	double f = 0.0;

	for (int m = 0; m < NUMBER_OF_DAMPERS; m++) {
		double w = wt[i + m * (K + 1)];
		double s = st[i + m * (K + 1)];
		double fNew = -x / (a * l) * (l - damperX[m] - s);
		if (x >= damperX[m] + s) {
			fNew += (x - damperX[m] - s) / a;
		}

		f += fNew * w;
	}

	return f;
}

__kernel void hessianAt(__global double* grad, __global double* h, __global double* wt, __global double* st, __global double* damperX, __global double* wtBounds, __global double* pu, double a, double l, double t, __global int* noFault) {
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

	cl_matrix2x2 sweepAlphaArr[N - 2];
	cl_matrix2x2 temp, invTemp;

	cl_matrix2x1 sweepBetaArr[N - 1];
	cl_matrix2x1 y0, y1, y2;
	cl_matrix2x1 temp1, temp2, temp3;
	cl_matrix2x1 currentV;

	double fA, fB, x, wA, wB, fNew, numInt = 0.0, dx, d0;
	double penR = 1.0, penC = 5.0;

	sweepAlphaArr[0].a = cInverted.a;
	sweepAlphaArr[0].b = cInverted.b;
	sweepAlphaArr[0].c = cInverted.c;
	sweepAlphaArr[0].d = cInverted.d;
	for (int i = 1; i < N - 2; i++) {
		temp.a = c.a - sweepAlphaArr[i - 1].a;
		temp.b = c.b - sweepAlphaArr[i - 1].b;
		temp.c = c.c - sweepAlphaArr[i - 1].c;
		temp.d = c.d - sweepAlphaArr[i - 1].d;

		detDenominator = temp.a * temp.d - temp.b * temp.c;
		sweepAlphaArr[i].a = temp.d / detDenominator;
		sweepAlphaArr[i].b = -temp.b / detDenominator;
		sweepAlphaArr[i].c = -temp.c / detDenominator;
		sweepAlphaArr[i].d = temp.a / detDenominator;
	}

	double u[N + 1];
	double v[N + 1];
	double f[N + 1];
	double bu[N + 1];
	double bwt[NUMBER_OF_DAMPERS * (K + 1)];
	double dst[NUMBER_OF_DAMPERS * (K + 1)];
	double pgrad[NUMBER_OF_DAMPERS * (K + 1)];
	double dampx[NUMBER_OF_DAMPERS];
	double dbounds[NUMBER_OF_DAMPERS * 2];

	for (int i = 0; i < N + 1; i++) {
		bu[i] = pu[i];
	}
	for (int i = 0; i < N + 1; i++) {
		u[i] = bu[i];
		v[i] = 0.0;
	}
	for (int i = 0; i < NUMBER_OF_DAMPERS; i++) {
		dampx[i] = damperX[i];

		//TODO fix memory issue, remove hardcoded values
		dbounds[i * 2] = -53.0;//wtBounds[0];
		dbounds[1 + i * 2] = 47.0;//wtBounds[1];
		for (int j = 0; j < K + 1; j++) {
			bwt[j + i * (K + 1)] = wt[j + i * (K + 1)];
			dst[j + i * (K + 1)] = st[j + i * (K + 1)];
			pgrad[j + i * (K + 1)] = grad[j + i * (K + 1)];
		}
	}

	double htemp[NUMBER_OF_DAMPERS * (K + 1)];

	for (int nth = 0; nth < NUMBER_OF_DAMPERS * (K + 1); nth++) {
		d0 = pgrad[nth];
		for (int j = 0; j < NUMBER_OF_DAMPERS * (K + 1); j++) {
			if (j >= nth) {
				bwt[j] += derH;
				for (int i = 1; i <= K; i++) {
					for (int j = 1; j < N; j++) {
						if (i == 1) {
							y0.a = bu[j - 1];
							y0.b = 0.0;

							y1.a = bu[j];
							y1.b = 0.0;

							y2.a = bu[j + 1];
							y2.b = 0.0;
						} else {
							y0.a = u[j - 1];
							y0.b = v[j - 1];

							y1.a = u[j];
							y1.b = v[j];

							y2.a = u[j + 1];
							y2.b = v[j + 1];
						}
				
						temp1.a = cTilde.a * y1.a + cTilde.b * y1.b;
						temp1.b = cTilde.c * y1.a + cTilde.d * y1.b;

						temp2.a = y0.a;
						temp2.b = y0.b;

						temp2.a -= temp1.a;
						temp2.b -= temp1.b;

						temp2.a += y2.a;
						temp2.b += y2.b;

						x = hx * j;
						fA = oscF(bwt, dst, dampx, a, l, x, i - 1);
						fB = oscF(bwt, dst, dampx, a, l, x, i);

						/*
						fA = 0.0;
						fB = 0.0;

						wA = bwt[i - 1];
						wB = bwt[i];

						fNew = -x * (1.0 - dampx[0]);
						if (x >= dampx[0]) {
							fNew += 1.0 * (x - dampx[0]);
						}

						fA += fNew * wA;
						fB += fNew * wB;
						*/

						currentV.a = -fA - fB;
						currentV.b = 0.0;

						currentV.a *= beta;
						currentV.b *= beta;

						temp2.a += currentV.a;
						temp2.b += currentV.b;

						if (j == 1) {
							sweepBetaArr[0].a = cInverted.a * temp2.a + cInverted.b * temp2.b;
							sweepBetaArr[0].b = cInverted.c * temp2.a + cInverted.d * temp2.b;
						} else {
							temp.a = c.a - sweepAlphaArr[j - 2].a;
							temp.b = c.b - sweepAlphaArr[j - 2].b;
							temp.c = c.c - sweepAlphaArr[j - 2].c;
							temp.d = c.d - sweepAlphaArr[j - 2].d;

							detDenominator = temp.a * temp.d - temp.b * temp.c;
							invTemp.a = temp.d / detDenominator;
							invTemp.b = -temp.b / detDenominator;
							invTemp.c = -temp.c / detDenominator;
							invTemp.d = temp.a / detDenominator;

							temp3.a = sweepBetaArr[j - 2].a;
							temp3.b = sweepBetaArr[j - 2].b;
							temp3.a += temp2.a;
							temp3.b += temp2.b;

							sweepBetaArr[j - 1].a = invTemp.a * temp3.a + invTemp.b * temp3.b;
							sweepBetaArr[j - 1].b = invTemp.c * temp3.a + invTemp.d * temp3.b;
						}
					}

					u[N - 1] = sweepBetaArr[N - 2].a;
					v[N - 1] = sweepBetaArr[N - 2].b;

					for (int j = N - 2; j > 0; j--) {
						u[j] = sweepAlphaArr[j - 1].a * u[j + 1] + sweepAlphaArr[j - 1].b * v[j + 1] + sweepBetaArr[j - 1].a;
						v[j] = sweepAlphaArr[j - 1].c * u[j + 1] + sweepAlphaArr[j - 1].d * v[j + 1] + sweepBetaArr[j - 1].b;
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

							dx = 0.0;
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

				numInt = f[0];
				for (int i = 1; i < N; i += 2) {
					numInt += 4.0 * f[i];
				}
				for (int i = 2; i < N; i += 2) {
					numInt += 2.0 * f[i];
				}
				numInt += f[N];
				numInt *= hx / 3.0;

				numInt += penalty(bwt, dst, dampx, dbounds, penR, penC);

				htemp[j] = numInt;

				bwt[nth] += derH;
				numInt = 0.0;
				for (int i = 1; i <= K; i++) {
					for (int j = 1; j < N; j++) {
						if (i == 1) {
							y0.a = bu[j - 1];
							y0.b = 0.0;

							y1.a = bu[j];
							y1.b = 0.0;

							y2.a = bu[j + 1];
							y2.b = 0.0;
						} else {
							y0.a = u[j - 1];
							y0.b = v[j - 1];

							y1.a = u[j];
							y1.b = v[j];

							y2.a = u[j + 1];
							y2.b = v[j + 1];
						}
				
						temp1.a = cTilde.a * y1.a + cTilde.b * y1.b;
						temp1.b = cTilde.c * y1.a + cTilde.d * y1.b;

						temp2.a = y0.a;
						temp2.b = y0.b;

						temp2.a -= temp1.a;
						temp2.b -= temp1.b;

						temp2.a += y2.a;
						temp2.b += y2.b;

						x = hx * j;
						fA = oscF(bwt, dst, dampx, a, l, x, i - 1);
						fB = oscF(bwt, dst, dampx, a, l, x, i);

						/*
						fA = 0.0;
						fB = 0.0;

						wA = bwt[i - 1];
						wB = bwt[i];

						fNew = -x * (1.0 - dampx[0]);
						if (x >= dampx[0]) {
							fNew += 1.0 * (x - dampx[0]);
						}

						fA += fNew * wA;
						fB += fNew * wB;
						*/

						currentV.a = -fA - fB;
						currentV.b = 0.0;

						currentV.a *= beta;
						currentV.b *= beta;

						temp2.a += currentV.a;
						temp2.b += currentV.b;

						if (j == 1) {
							sweepBetaArr[0].a = cInverted.a * temp2.a + cInverted.b * temp2.b;
							sweepBetaArr[0].b = cInverted.c * temp2.a + cInverted.d * temp2.b;
						} else {
							temp.a = c.a - sweepAlphaArr[j - 2].a;
							temp.b = c.b - sweepAlphaArr[j - 2].b;
							temp.c = c.c - sweepAlphaArr[j - 2].c;
							temp.d = c.d - sweepAlphaArr[j - 2].d;

							detDenominator = temp.a * temp.d - temp.b * temp.c;
							invTemp.a = temp.d / detDenominator;
							invTemp.b = -temp.b / detDenominator;
							invTemp.c = -temp.c / detDenominator;
							invTemp.d = temp.a / detDenominator;

							temp3.a = sweepBetaArr[j - 2].a;
							temp3.b = sweepBetaArr[j - 2].b;
							temp3.a += temp2.a;
							temp3.b += temp2.b;

							sweepBetaArr[j - 1].a = invTemp.a * temp3.a + invTemp.b * temp3.b;
							sweepBetaArr[j - 1].b = invTemp.c * temp3.a + invTemp.d * temp3.b;
						}
					}

					u[N - 1] = sweepBetaArr[N - 2].a;
					v[N - 1] = sweepBetaArr[N - 2].b;

					for (int j = N - 2; j > 0; j--) {
						u[j] = sweepAlphaArr[j - 1].a * u[j + 1] + sweepAlphaArr[j - 1].b * v[j + 1] + sweepBetaArr[j - 1].a;
						v[j] = sweepAlphaArr[j - 1].c * u[j + 1] + sweepAlphaArr[j - 1].d * v[j + 1] + sweepBetaArr[j - 1].b;
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

							dx = 0.0;
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

				numInt = f[0];
				for (int i = 1; i < N; i += 2) {
					numInt += 4.0 * f[i];
				}
				for (int i = 2; i < N; i += 2) {
					numInt += 2.0 * f[i];
				}
				numInt += f[N];
				numInt *= hx / 3.0;

				numInt += penalty(bwt, dst, dampx, dbounds, penR, penC);

				htemp[j] = (numInt - htemp[j]) / derH;
				htemp[j] = (htemp[j] - d0) / derH;

				bwt[nth] -= derH;
				bwt[j] -= derH;
			}
		}
		for (int j = 0; j < NUMBER_OF_DAMPERS * (K + 1); j++) {
			if (j >= nth) {
				h[j + (K + 1) * nth] = htemp[j];
			}
		}
	}

	noFault[0] = 1;
}

