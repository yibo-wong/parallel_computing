parallel: homework.cpp
	mpic++ -D __MPI -o parallel homework.cpp
serial: homework.cpp
	g++ homework.cpp -o serial


clean:
	rm -f serial parallel
	rm -rf output