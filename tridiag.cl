#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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

cl_matrix2x2 sweepAlpha(cl_matrix2x2 sweepAlphaArr, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 sweepSide, int i) {
	cl_matrix2x2 alpha;
	if (i == 0) {
		alpha = matrixMultiply2x2(cInverted, sweepSide);
	} else {
		cl_matrix2x2 temp = c;
		cl_matrix2x2 temp2 = matrixMultiply2x2(sweepSide, sweepAlphaArr);
		temp.a -= temp2.a;
		temp.b -= temp2.b;
		temp.c -= temp2.c;
		temp.d -= temp2.d;

		cl_matrix2x2 invTemp = matrixInverse(temp);

		alpha = matrixMultiply2x2(invTemp, sweepSide);
	}
	return alpha;
}

cl_matrix2x1 sweepBeta(cl_matrix2x2 sweepAlphaArr, cl_matrix2x1 sweepBetaArr, cl_matrix2x1 right, cl_matrix2x2 c, cl_matrix2x2 cInverted, cl_matrix2x2 sweepSide, int i) {
	cl_matrix2x1 beta;
	if (i == 0) {
		beta = matrixMultiply(cInverted, right);
	} else {
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
	}
	return beta;
}

__kernel void tridiag(__global cl_matrix2x1* x, __global cl_matrix2x1* right, __global cl_matrix2x2* c_arr, __global cl_matrix2x2* cInverted_arr, __global cl_matrix2x2* sweepSide_arr, int n, __global int* no_fault) {
	cl_matrix2x2 c = c_arr[0];
	cl_matrix2x2 cInverted = cInverted_arr[0];
	cl_matrix2x2 sweepSide = sweepSide_arr[0];

	cl_matrix2x2 sweepAlphaArr[14];
	for (int i = 0; i < 14; i++) {
		sweepAlphaArr[i] = sweepAlpha(sweepAlphaArr[i - 1], c, cInverted, sweepSide, i);
	}

	cl_matrix2x1 sweepBetaArr[15];
	for (int i = 0; i < 15; i++) {
		sweepBetaArr[i] = sweepBeta(sweepAlphaArr[i - 1], sweepBetaArr[i - 1], right[i], c, cInverted, sweepSide, i);
	}

	x[14] = sweepBetaArr[14];
	for (int i = 13; i >= 0; i--) {
		cl_matrix2x1 temp = matrixMultiply(sweepAlphaArr[i], x[i + 1]);
		temp.a += sweepBetaArr[i].a;
		temp.b += sweepBetaArr[i].b;
		x[i] = temp;
	}

	no_fault[0] = 1;
}

