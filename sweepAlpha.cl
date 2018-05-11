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

__kernel void sweepAlpha(double a, double l, double t, __global cl_matrix2x2* sweepAlphaArr) {
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

	sweepAlphaArr[0].a = cInverted.a;
	sweepAlphaArr[0].b = cInverted.b;
	sweepAlphaArr[0].c = cInverted.c;
	sweepAlphaArr[0].d = cInverted.d;
	for (int i = 1; i < N - 2; i++) {
		cl_matrix2x2 temp;
		temp.a = c.a - sweepAlphaArr[i - 1].a;
		temp.b = c.b - sweepAlphaArr[i - 1].b;
		temp.c = c.c - sweepAlphaArr[i - 1].c;
		temp.d = c.d - sweepAlphaArr[i - 1].d;

		double detDenominator = temp.a * temp.d - temp.b * temp.c;
		sweepAlphaArr[i].a = temp.d / detDenominator;
		sweepAlphaArr[i].b = -temp.b / detDenominator;
		sweepAlphaArr[i].c = -temp.c / detDenominator;
		sweepAlphaArr[i].d = temp.a / detDenominator;
	}
}

