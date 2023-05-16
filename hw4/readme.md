# 并行程序设计 计时器作业 
>2100011025 工学院 王奕博

## 编译说明

在当前路径下，使用
```
make
```
进行编译，之后使用
```
./timer
```
查看结果。

## 程序架构
程序的主要架构由record类和timer类构成。record类包含四个成员变量和一些成员函数，成员变量包括类名、函数名、被调用次数以及vector类型的每次被调用的时间，成员函数包括了一些简单的求和、求平均等功能，以及添加新的调用记录等。

它的具体构造如下:
```c
class record
{
public:
        string class_name;
        string func_name;
        int times;
        vector<double> history;// the records of duration of a certain function
        record(string c,string f):class_name(c),func_name(f)
        {
                times = 0;
                history.clear();
        }
        ~record(){}
        void add(double record)// add a new record to the history
        {
                times += 1;
                history.push_back(record);
        }
        double total()// calculate total time
        {...}
        double average()// calculate average time
        {...}
};
```

timer类则主要由一些静态成员函数和静态变量构成，这使得在使用timer类的时候无需实例化某个特定的timer类，直接调用相关函数即可。

timer类的记录是由一个储存record类的vector实现的，即：
```c
static vector<record> records;// store the record class
```

timer类用std库中的map来记录已经出现过的函数名和类名，这里使用pair将两个string并为整体，并使用
```c
static map<pair<string, string>, int> entry;
```
来记录以及防止重复。

同时，timer类使用chrono库中的相关功能来实现计时器的功能，定义的变量如下：
```c
static chrono::time_point<chrono::steady_clock> start_time;
```
此外，还使用了一个bool类型的flag，每当使用一次tick函数时，flag会从0变成1或者从1变成0，用来开启或者关闭计时器。

主要的tick函数控制计时功能，它的结构如下：
```c
static void tick(const string& class_name, const string& function_name)
        {
                        start_time = chrono::steady_clock::now();
                {
                        flag = 1;
                        start_time = chrono::steady_clock::now();
                }
                else if (flag == 1)
                {
                        chrono::duration<float> chrono_duration = chrono::steady_clock::now() - start_time;
                        double duration = chrono_duration.count();

                        pair<string, string> name;
                        name = make_pair(class_name, function_name);
                        map<pair<string, string>, int>::iterator it;
                        it = entry.find(name);
                        if (it == entry.end())// if the function have not appeared
                        {
                                record new_record = record(class_name,function_name);
                                new_record.add(duration);
                                int index = entry.size();
                                entry[name] = index;
                                records.push_back(new_record);
                        }
                        else// if it has appeared
                        {
                                int index = it->second;
                                records[index].add(duration);
                        }
                        total_time += duration;
                        flag = 0;
                }
        }
```
如果是第一次打开，tick函数会开始计时；第二次打开时，tick函数立即关闭计时器（为了防止其它操作也被计入计时），之后确定被计时的函数是否已经被记录过。如果被记录则找到那条record类，向其中加入这条记录；如果没有，则加入新的record类。

## 测试结果

在这里，我想测试cmath库中exp(),sin(),log()哪一种函数消耗的时间最长。我分别计算$\sum_{i=1}^{200} exp(i)$,$\sum_{i=1}^{200} sin(i)$,$\sum_{i=1}^{200} log(i)$的值，这其中每一种函数都被调用了200次。

测试的Main函数如下：
```c
int main(int argc,char** argv)
{
        double y;
        // here's my test for the average time consuming of three functions,log(x),exp(x) and sin(x). 
        // the result shows sin(x) is the slowest of three all.
        // emmm...sometimes log(x) takes the most time. i dont know why :( 
        y = 1.0;
        for (int i = 1; i <= 200; i++)
        {
                timer::tick("none", "log");
                y = y + log(i);
                timer::tick("none", "log");
        }

        y = 1.0;
        for (int i = 1; i <= 200; i++)
        {
                timer::tick("none", "exp");
                y = y + exp(i);
                timer::tick("none", "exp");
        }

        y = 1.0;
        for (int i = 1; i <= 200; i++)
        {
                timer::tick("none", "sin");
                y = y + sin(i);
                timer::tick("none", "sin");
        }

        return 0;
}
```

编译之后，几次结果为：
```
|CLASS_NAME-----|NAME----------|TIME(sec)-----|CALLS---------|AVG-----------|PER%----------|
          TOTAL          TOTAL     7.0359e-05                                          100 %
           none            log     2.7429e-05            200    1.37145e-07        38.9844 %
           none            exp     1.6196e-05            200      8.098e-08        23.0191 %
           none            sin     2.6734e-05            200     1.3367e-07        37.9966 %  
```

```
|CLASS_NAME-----|NAME----------|TIME(sec)-----|CALLS---------|AVG-----------|PER%----------|
          TOTAL          TOTAL       6.93e-05                                          100 %
           none            log     2.6329e-05            200    1.31645e-07        37.9928 %
           none            exp     1.6203e-05            200     8.1015e-08         23.381 %
           none            sin     2.6768e-05            200     1.3384e-07        38.6263 %
```

```
|CLASS_NAME-----|NAME----------|TIME(sec)-----|CALLS---------|AVG-----------|PER%----------|
          TOTAL          TOTAL     8.5729e-05                                          100 %
           none            log     2.6346e-05            200     1.3173e-07        30.7317 %
           none            exp     2.0504e-05            200     1.0252e-07        23.9172 %
           none            sin     3.8879e-05            200    1.94395e-07         45.351 %
```

故结果是，log和sin占用的时间比exp长。

## 一些不足

一个尚未解决的不足之处：

在这里，如果我将计时器放在for循环外而非循环内，占用的总时间会少大约30%，这表明从调用tick()函数到计时被中止（其实只是调用开销+一行if判断，之后的时间并不计入计时），所占用的时间和调用log()等函数基本在同一个数量级内。我试图优化了一小部分，但是没有显著的方法减少这个时间。

其实这个时间并不长，不过为了这个tick()能够更好的工作，应该尽量使得每次调用的间隔长一点，这样误差会更小。