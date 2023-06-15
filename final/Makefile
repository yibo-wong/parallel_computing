omp: omp_v0.cpp
	g++ -fopenmp omp_v0.cpp -o omp
serial: serial_v1.cpp
	g++ serial_v1.cpp -o serial
serial_origin: serial_v0.cpp
	g++ serial_v0.cpp -o serial_origin
diago: diago.cpp
	mpic++ diago.cpp -lscalapack-openmpi -o diago

clean:
	rm -f serial omp serial_origin diago a.out
	rm -f ./result/hamilton.txt ./result/diago.txt