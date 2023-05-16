// Yibo Wang, 2100011025, coe_pku, parallel_programming.
#include <iostream>
#include <string>
#include <cstring>
#include <chrono>
#include <vector>
#include <map>
#include <iomanip>
#include <cmath>

using namespace std;

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
	{
		if (history.empty()) return 0.0;
		else
		{
			double sum = 0;
			for (int i = 0; i < history.size(); i++) sum += history[i];
			return sum;
		}
	}
	double average()// calculate average time
	{
		if (history.empty()) return 0.0;
		else
		{
			double sum = 0;
			for (int i = 0; i < history.size(); i++) sum += history[i];
			return sum / times;
		}
	}
};

class timer
{
public:
	// most variables are static, allowing the timer working in a static member function's way
	static double total_time;
	timer(){}
	~timer(){}
	static map<pair<string, string>, int> entry;// record the class names and the functions names
	static vector<record> records;// store the record class
	static bool flag;// a 'switch' that can be turned on and turned off
	static chrono::time_point<chrono::steady_clock> start_time;// chrono library
	static void tick(const string& class_name, const string& function_name)
	{
		if (flag == 0)
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
	static void print()
	{
		cout << "|CLASS_NAME-----|NAME----------|TIME(sec)-----|CALLS---------|AVG-----------|PER%----------|" << endl;

		cout << setw(15);
		cout << "TOTAL";
		cout << setw(15);
		cout << "TOTAL";
		cout << setw(15);
		cout << total_time;
		cout << setw(47);
		cout << "100 %";
		cout << endl;

		for (int i = 0; i < records.size(); i++)
		{
			double portion = records[i].total() / total_time;
			cout << setw(15);
			cout << records[i].class_name;
			cout << setw(15);
			cout << records[i].func_name;
			cout << setw(15);
			cout << records[i].total();
			cout << setw(15);
			cout << records[i].times;
			cout << setw(15);
			cout << records[i].average();
			cout << setw(15);
			cout << portion * 100 << " %";
			cout << endl;
		}
	}
};

double timer::total_time = 0.0;
map<pair<string, string>, int> timer::entry;
vector<record> timer::records;
bool timer::flag = 0;
chrono::time_point<chrono::steady_clock> timer::start_time;

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

	timer::print();
	return 0;
}
