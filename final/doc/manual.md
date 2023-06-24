# 使用文档: openmp / cuda
>2100011025 王奕博

**写在前面：我确信我的程序都是可以正确运行的，然而文字描述如何运行难免出现纰漏。因此，如果真的遇到运行不了的情况，请求助教联系我，共同解决问题，非常感谢~**

## 0. 准备操作

在final目录下的input目录中，有文件 **/final/input/INPUT_test.txt** 。

这是整体的输入文件，包含了（其实只有这些是有用的）：lx,ly,lz长度，以及其余三个读入文件的路径（**注意：是相对于final文件夹的！也就是说，“.”指的是final而非input**）。另外，此文件支持注释，不过请按照
```
这不是注释
# 这是注释（开头#再加空格）
```
的格式进行注释。

里面有测试路径，但是由于几个测试文件大小过大，故不传上去，烦请助教自行操作。此处建议按照：
```
├── input
│   ├── INPUT_test.txt
│   ├── correctness
│   │   ├── case1
│   │   │   ├── V-1.txt
│   │   │   ├── distribution80.txt
│   │   │   └── point2.txt
│   │   ├── case2
│   │   │   ├── V-1.txt
│   │   │   ├── distribution80.txt
│   │   │   └── point2.txt
│   │   └── case3
│   │       ├── V-1.txt
│   │       ├── distribution80.txt
│   │       └── point2.txt
│   └── efficiency
│       ├── Distribution
│       │   └── distribution80.txt
│       ├── Point
│       │   ├── point10.txt
│       │   ├── point2.txt
│       │   └── point50.txt
│       └── V
│           ├── V128.txt
│           ├── V256.txt
│           └── V512.txt
```
这种路径放置，correctness内部为正确性测试文件，efficiency内部为效率测试文件。

还有：**由于之前在群里发的point文件格式有不一致的情况，故请确定point文件的格式是"(a,b,c)"而非"a b c"！** 否则，这很容易将整个程序搞崩溃。

此外，我默认测试机器是安装好lapack以及scalapack的，即可以直接引用lapacke.h。scalapack是完全按照韩助教的指示安装的，应该问题不大。

## 1. 串行版本

目录下有serial_v0.cpp以及serial_v1.cpp。其中，v0为非常非常原始的版本，无任何优化，不建议动它。

如果运行优化过的串行程序,首先编译命令：

```
make serial
```

之后运行命令：

```
./serial
```
运行时会在屏幕上显示进程，结束后会打印计时器，calc项是计算所用时间。

这会在result文件夹下输出所求的hamilton矩阵hamilton.txt，可以查看。注意， **对角化程序是单独写的，这里并不包含对角化步骤**。

如果使用完毕，请使用命令
```
make clean
```
进行清理，注意这会**同时删除输出的txt文件**,因此如果要继续对角化的话，请暂时不要清理。

## 2.并行（openmp）版本

编译命令：
```
make omp
```

运行命令：
```
./omp
```

之后，会在result文件夹下输出所求的hamilton矩阵hamilton.txt，屏幕上也会显示时长，calc项是计算所用时间。

**如果之后还要对角化，请不要清理**；否则，使用
```
make clean
```
清理。

## 3.并行（cuda）版本

在提供的GPU测试机器上，使用：
```
make cuda
```
进行编译。中间可能会提示一大堆warning，但是它们都不影响运行和结果。

之后使用：
```
./cuda
```
运行程序。屏幕上会显示进程，以及纯粹计算时间所消耗的时长（以毫秒为单位）。

之后，会在result文件夹下输出所求的hamilton矩阵hamilton.txt。

**如果之后还要对角化，请不要清理**；否则，使用
```
make clean
```
清理。

## 4.对角化（lapack）

**对角化之前，请确认result文件夹下有hamilton.txt文件。这很重要！**

使用lapack对角化，即串行版本的对角化，需要编译程序：
```
make diago_serial
```
运行程序:
```
./diago_serial
```
程序会读取hamilton.txt，并将对角化结果（包括特征值和特征向量）输入到diago_lapack.txt中，可以查看结果。

## 5.对角化（scalapack）

使用并行版本的对角化，编译命令（当然，请确保测试机器上是有相关环境的）：
```
make diago_parallel
```
运行命令：**这里使用mpirun，请确保一定是4核运行的。** 我在程序里支持了其它核数，但是改起来比较麻烦。如果使用root用户运行，还要加上--allow-run-as-root.
```
mpirun -np 4 ./diago_parallel
```

程序会将结果放在diago_scalapack.txt中。

以上便是全部程序功能，具体程序实现细节请见代码报告。