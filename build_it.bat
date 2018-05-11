mkoctfile gpuMarquardtMinimization.cc hyperb.cc sweepAlpha.cc energyInt.cc gradient.cc hessian.cc -L"." -lOpenCL -pedantic -Wall
mkoctfile gpuEnergyInt.cc hyperb.cc sweepAlpha.cc energyInt.cc -L"." -lOpenCL -pedantic -Wall
mkoctfile gpuOscDeadening.cc hyperb.cc sweepAlpha.cc energyInt.cc -L"." -lOpenCL -pedantic -Wall