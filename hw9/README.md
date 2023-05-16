# 并行程序设计 MPI矩阵加法
 
>2100011025 工学院 王奕博

## 编译说明

在主目录下有一个INPUT.txt文档，其中包括了一些输入内容。我将教学网上的matadd测试文件也包含在文件夹中，这样，将INPUT.txt中的case_3改为case_1到case5，就能测试matadd中的五组文件。

### 串行

如果要编写**串行**程序，则通过命令
```
make serial
```
进行编译。

之后，使用命令
```
./serial
```
查看结果。时间会打印在屏幕上，记录文件在output/目录下。

### 并行

如果要编写**并行**程序，则通过命令
```
make parallel
```
进行编译。

之后，使用命令
```
mpirun --oversubscribe --allow-run-as-root -np 4 ./parallel
```
运行并查看结果（这里展示的是6核的结果，用别的核数只需要把4改成核数即可）（**如果不是root用户，则不用加上--allow-run-as-root**）。时间会打印在屏幕上，记录文件在./output/目录下。

(顺便说一下发现的一个问题，在一些机器上有时第一次运行的时候MPI_barrier不起作用，第二次运行时就好了。)

### 重新编译

如果需要重新编译的话，可以使用
```
make clean
```
来删除相关文件，之后可以重新编译。具体来说，它删除了parallel,serial,以及./output下的文件。

## 目录管理

```
.
├── INPUT.txt
├── Makefile
├── homework.cpp
├── include
│   ├── input.h
│   ├── matrix.h
│   └── timer.h
├── matadd
│   ├── case_1
│   │   ├── mat_A.txt
│   │   └── mat_B.txt
│   ├── case_2
│   │   ├── mat_A.txt
│   │   └── mat_B.txt
│   ├── case_3
│   │   ├── mat_A.txt
│   │   └── mat_B.txt
│   ├── case_4
│   │   ├── mat_A.txt
│   │   └── mat_B.txt
│   └── case_5
│       ├── mat_A.txt
│       └── mat_B.txt
├── README.pdf
└── README.md
```

主目录下包含了输入文件INPUT.txt，源文件homework.cpp，makefile文件以及readme文件。

头文件放在目录./include中，包含input.h,matrix.h,timer.h。

./matadd存放的是矩阵输入文件，通过更改INPUT.txt中的内容来测试。

## 程序思路

本次作业程序按照以下逻辑来运行：

### 1
首先，编号0的进程读入文件，并将一些全局的变量通过MPI_Bcast函数传给其它进程，这些包括：
```c
 // broadcast alpha,beta,print_log,timer_print,output_c,total_rows,col_num
MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&print_log, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&timer_print, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(output_c, max_string_length, MPI_CHAR, 0, MPI_COMM_WORLD);
MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
MPI_Bcast(&col_num, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

### 2

在这之后，主进程（编号为0的进程）通过行来划分矩阵，然后分配给其它进程。

具体来说，主进程计算出每个进程对应的起始位置与行数，然后将指针平移到该位置，用MPI_send发送数据，对应的进程接收。

主要代码如下：

```c
for (int i = 1; i < world_size; i++) {
      int rank = i;
      int ave = total_rows / world_size;
      int this_row_num = (rank < total_rows % world_size) ? ave + 1 : ave;
      int row_start = (rank < total_rows % world_size)
                          ? rank * (ave + 1)
                          : (total_rows % world_size) * (ave + 1) +
                                (rank - total_rows % world_size) * ave;

      send_data_a = input_a->element + row_start * col_num;
      send_data_b = input_b->element + row_start * col_num;
      MPI_Send(&row_start, 1, MPI_INT, rank, 1, MPI_COMM_WORLD);
      MPI_Send(&this_row_num, 1, MPI_INT, rank, 2, MPI_COMM_WORLD);
      MPI_Send(send_data_a, this_row_num * col_num, MPI_DOUBLE, rank, 3,
               MPI_COMM_WORLD);
      MPI_Send(send_data_b, this_row_num * col_num, MPI_DOUBLE, rank, 4,
               MPI_COMM_WORLD);
    }
```

### 3

传输完成之后，每个进程计算加法。由于在matrix.h里已经做好了运算符重载，故可以直接计算：

```c
mat_result = (mat_a * alpha) + (mat_b * beta);
```

### 4
之后，每个进程将计算结果发回主进程，主进程读入并写入最终结果矩阵中。

相关代码如下：
```c
final_result.init_partly(mat_result.element, 0, row_num * col_num);
    for (int i = 1; i < world_size; i++) {
      int rank = i;
      int ave = total_rows / world_size;
      int this_row_num = (rank < total_rows % world_size) ? ave + 1 : ave;
      int row_start = (rank < total_rows % world_size)
                          ? rank * (ave + 1)
                          : (total_rows % world_size) * (ave + 1) +
                                (rank - total_rows % world_size) * ave;
      double *recv_data = new double[this_row_num * col_num];
      MPI_Recv(recv_data, this_row_num * col_num, MPI_DOUBLE, i, 5,
               MPI_COMM_WORLD, &status);
      final_result.init_partly(recv_data, row_start * col_num,
                               col_num * this_row_num);
    }
```
其中，init_partly()是matrix.h中的函数，作用是将数组写入矩阵的对应位置。

### 5
最后，主进程将结果写入文件，然后每个进程轮次在屏幕上打印时间（通过MPI_barrier控制次序）。

## 运行结果

矩阵加法结果可以在./output目录下查看，可以成功计算矩阵加法。

在我的机器上运行时，对于某次运行，时间打印如下：
```
IN WORLD 0 :
|CLASS_NAME-----|NAME----------|TIME(sec)-----|CALLS---------|AVG-----------|PER%----------|
          TOTAL          TOTAL      0.0290019                                          100 %
     main: read         matrix       0.008568              1       0.008568        29.5429 %
     main: send           data        4.5e-05              1        4.5e-05       0.155162 %
           calc            add      0.0002253              1      0.0002253       0.776846 %
     main: recv           data      0.0001041              1      0.0001041       0.358942 %
    main: write           file      0.0200595              1      0.0200595        69.1662 %

IN WORLD 1 :
|CLASS_NAME-----|NAME----------|TIME(sec)-----|CALLS---------|AVG-----------|PER%----------|
          TOTAL          TOTAL      0.0100454                                          100 %
    other: recv           data      0.0091473              1      0.0091473        91.0596 %
           calc            add       0.000148              1       0.000148        1.47331 %
    other: send           data      0.0007501              1      0.0007501         7.4671 %

IN WORLD 2 :
|CLASS_NAME-----|NAME----------|TIME(sec)-----|CALLS---------|AVG-----------|PER%----------|
          TOTAL          TOTAL      0.0101797                                          100 %
    other: recv           data       0.009298              1       0.009298        91.3386 %
           calc            add      0.0001327              1      0.0001327        1.30357 %
    other: send           data       0.000749              1       0.000749        7.35778 %

IN WORLD 3 :
|CLASS_NAME-----|NAME----------|TIME(sec)-----|CALLS---------|AVG-----------|PER%----------|
          TOTAL          TOTAL      0.0102396                                          100 %
    other: recv           data      0.0094138              1      0.0094138        91.9352 %
           calc            add      0.0002068              1      0.0002068        2.01961 %
    other: send           data       0.000619              1       0.000619        6.04516 %
```
可以看出，对于主进程，绝大多数时间用于从文件中读写，而对于其它进程，绝大多数时间用于MPI_recv函数接收发来的数据。
