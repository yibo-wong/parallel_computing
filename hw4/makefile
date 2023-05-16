CC = g++
TARGET = timer
OBJK = timer.o

$(TARGET): $(OBJK)
	$(CC) -o $@ $^

timer.o:timer.cpp
	$(CC) -c timer.cpp -o $@
.PHONY:clean
clean:
	rm -f *.o  $(TARGET)
