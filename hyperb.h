#ifndef HYPERB_H
#define HYPERB_H

typedef struct alphastruct {
    cl_double a, b, c, d;
} cl_matrix2x2;

typedef struct betastruct {
    cl_double a, b;
} cl_matrix2x1;

// hyperb.cc
double* toDouble(Matrix u, int n, int k);
Matrix toMatrix(double* u, const int n, const int k);
void printGrid(double* u, int n, int k);

// sweepAlpha.cc
void createAndBuildSweepAlphaKernel();
cl_matrix2x2* solveAlpha(cl_double* damperX, cl_double* wt);
void releaseSweepAlphaKernel();

// energyInt.cc
void createAndBuildKernel();
cl_double solveEnergyInt(cl_matrix2x2* sweepAlphaArr, cl_double* damperX, cl_double* wt, Matrix wtBounds, Matrix st, cl_double* u, cl_double* v, bool includePenalty);
void releaseKernel();

// gradient.cc
void createAndBuildGradientKernel();
Matrix gpuGradientAt(cl_matrix2x2* sweepAlphaArr, cl_double sei, Matrix wt, Matrix st, double derH, cl_double* dampx, Matrix wtBounds, cl_double* u, cl_double* v);
void releaseGradientKernel();
void printGradient(Matrix gradient, int iter, int m);
double euclideanGradientNorm(Matrix gradient, int k);

// hessian.cc
void createAndBuildHessianKernel(cl_matrix2x2* sweepAlphaArr, Matrix wt, Matrix st, cl_double derH, cl_double* dampx, Matrix wtBounds, cl_double* du);
Matrix gpuHessianAt(cl_matrix2x2* sweepAlphaArr, Matrix gradient, Matrix wt, Matrix st, double derH, double* dampx, Matrix wtBounds, cl_double* u, cl_double* v, bool isDebug);
void releaseHessianKernel();
void printHessianMatrix(Matrix h, int iter, int m);
double euclideanHessianNorm(Matrix h, int k);

#endif // HYPERB_H
