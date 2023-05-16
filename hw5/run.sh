#!/bin/bash

g++ mat_all.cpp -o mat_all -llapacke -llapack -lcblas  -lrefblas  -lm -lgfortran

./mat_all
