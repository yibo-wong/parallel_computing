# 并行程序设计 MPI矩阵数据传输
 
>2100011025 工学院 王奕博

## 编译说明

**本次作业没有input文档，直接对矩阵输入文件进行操作。**

**程序从test.txt中读入数据，然后进行拆分之类的操作。如果需要测试mat-32-16.txt的数据，则需删除test.txt同时将mat-32-16.txt改名为test.txt。**

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
mpirun --allow-run-as-root -np 6 ./parallel
```
运行并查看结果（这里展示的是6核的结果，用别的核数只需要把6改成核数即可）（**如果不是root用户，则不用加上--allow-run-as-root**）。时间会打印在屏幕上，记录文件在output/目录下。

### 重新编译

如果需要重新编译的话，可以使用
```
make clean
```
来删除相关文件，之后可以重新编译。具体来说，它删除了parallel,serial,以及全部的输出文件。

## 程序架构

本次程序较为简单，首先是改写了timer类，让其可以在并行环境下运行。

具体的更改如下：
```c
#ifdef __MPI
    if (flag == 0) {
      flag = 1;
      start_time_mpi = MPI_Wtime();
      return;
    } else if (flag == 1) {
      double endtime = MPI_Wtime();
      duration = endtime - start_time_mpi;
    }
#else
    if (flag == 0) {
      flag = 1;
      start_time = chrono::steady_clock::now();
      return;
    } else if (flag == 1) {
      chrono::duration<float> chrono_duration =
          chrono::steady_clock::now() - start_time;
      duration = chrono_duration.count();
    }
#endif
```

对于主要程序，则是首先读入矩阵类，然后通过计算MPI_Comm_rank判断具体读入哪些元素。具体如下：
```c
  int ave = total_rows / world_size;
  int row_num = (world_rank < total_rows % world_size) ? ave + 1 : ave;
  int row_start = (world_rank < total_rows % world_size)
                      ? world_rank * (ave + 1)
                      : (total_rows % world_size) * (ave + 1) +
                            (world_rank - total_rows % world_size) * ave;

  Matrix mat_mpi(row_num, col_num);
  mat_mpi.init(input->element, row_start * col_num);
```
其中，init函数是Matrix下的一个成员函数，它读入某个地址与某个长度len，然后将这个地址之后的len个元素赋值给Matrix本身。

