omp: omp_v0.cpp
	g++ -fopenmp omp_v0.cpp -o omp
serial: serial_v1.cpp
	g++ serial_v1.cpp -o serial
serial_origin: serial_v0.cpp
	g++ serial_v0.cpp -o serial_origin
diago_parallel: diago.cpp
	mpic++ diago.cpp -lscalapack-openmpi -o diago_parallel
diago_serial: diago_serial.cpp
	g++ diago_serial.cpp -o diago_serial -llapacke -llapack -lm -lgfortran
cuda: cuda_v0.cu
	nvcc cuda_v0.cu -o cuda

clean:
	rm -f serial omp serial_origin diago_serial diago_parallel a.out