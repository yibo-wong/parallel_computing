TARGET=mat_all
LIBS=-llapacke -llapack -lcblas  -lrefblas  -lm -lgfortran
CC=g++

$(TARGET):
	$(CC) mat_all.cpp -o $(TARGET) $(LIBS)

clean:
	$(RM) $(TARGET)