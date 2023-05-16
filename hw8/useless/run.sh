#!/bin/bash

mpic++ -D __MPI homework.cpp -o parallel

mpirun --allow-run-as-root -np 4 ./parallel
