# 并行程序设计 计时器作业 
>2100011025 工学院 王奕博

## 编译说明

**首先，需求读入的文档是：**
```
demand.txt
```
***可以按照作业要求修改这个文档的内容，来测试不同的功能。*** **比如，将matmul修改成RSMdiago，以测试矩阵相乘功能。**

同时，demand.txt中的默认文档位置是./diago/case_3/mat_1.txt，**这需要将教学网上那个测试文件放入这个文件夹中**。由于那两个文件比较大，我就不一起传上来了，劳烦助教完成。

**在当前路径下，使用**
```
make
```
**进行编译，之后使用**
```
./mat_all
```
**查看结果。**

**如果需要重新编译的话，可以使用**
```
make clean
```
**来删除相关文件，之后可以重新编译。**

## 程序架构

本次作业由多文件构成，包括主程序：
```
mat_all.cpp
```
包含输入类的.h文件：
```
input.h
```
包含基础矩阵类的.h文件：
```
matrix.h
```

以及用float,complex<float>,complex<double>的矩阵类头文件：
```
matrix_float.h
matrix_com_double.h
matrix_com_float.h
```

包含计时类的文件
```
timer.h
```
以及makefile文件构成。

其中，输入类、矩阵类、计时类都是基于之前的代码修改的，故不再叙述它们的架构。其中，我做了以下修改：

1. 将矩阵元素的存储从double** 改为了double*，这是为了和cblas中的矩阵运算相适应；

2. 输入类针对作业要求做了相关修改。

对于矩阵类，我按照作业要求加入了一些函数，如下：

### 重载*符号实现矩阵乘法

```c
Matrix operator*(Matrix &A, Matrix &B) {
  int r = A.getRows();
  int c = B.getCols();
  int p = A.getCols();
  Matrix result = Matrix(r, c);
  if (A.getCols() != B.getRows()) {
    cout << "wrong dimension" << endl;
    return result;
  }
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      double sum = 0;
      for (int k = 0; k < p; k++) {
        sum += A(i, k) * B(k, j);
      }
      result(i, j) = sum;
    }
  }
  return result;
}
```

### 使用cblas_dgemm进行矩阵乘法

```c
friend Matrix blas_matmul(Matrix &A, Matrix &B) {
    int m = A.nrows;
    int n = B.ncols;
    int k = A.ncols;
    double alpha = 1.0;
    double beta = 0.0;
    double *matA = A.element;
    double *matB = B.element;
    double *matC = new double[m * n];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, matA, k, matB, n, beta, matC, n);
    Matrix C(m, n);
    C.init(matC);
    return C;
  }
```

### 判断矩阵是否对称

```c
bool isRSM() {
    if (nrows != ncols)
      return 0;
    bool flag = 1;
    for (int i = 0; i < nrows; i++) {
      for (int j = i + 1; j < ncols; j++) {
        if (element[i * ncols + j] != element[j * ncols + i])
          return 0;
      }
    }
    return 1;
  }
```

### 用dgeev进行对角化

```c
friend int RSMdiago(Matrix &A, Matrix &wr, Matrix &wi, Matrix &vr, Matrix &vl) {
    if (!A.isRSM()) {
      return 1;
    }
    int n = A.getRows();
    int lda = n;
    int ldvl = n;
    int ldvr = n;
    int info = 0;
    int lwork = 0;
    char jobvl = 'V';
    char jobvr = 'V';

    LAPACKE_dgeev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, A.element, lda, wr.element, wi.element, vl.element, ldvl, vr.element, ldvr);

    return info;
  }
```

## 附加题
在头文件
```
matrix_float.h
matrix_com_double.h
matrix_com_float.h
```
中，我将矩阵类型分别换成了float,complex<float>,complex<double>，并调用对应的cblas中的函数（分别是cblas_sgemm,cblas_cgemm,cblas_zgemm，对应单精度、单精度浮点数、多精度浮点数）。（这是仅有的两处修改，其他地方完全相同，除了删除了多余的函数）

虽然没来得及更新对应的读入文件，但我通过terminal输入测试了这些文件的正确性。