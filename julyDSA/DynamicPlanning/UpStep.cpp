/* 题目：上台阶
有一个楼梯总共有n个台阶，只能往上走，每次只能上1个或2个台阶，总共有多少种走法。
 */

#include <iostream>
#include <vector>
#include <map>
using namespace std;

map<int, int> cache;
int upStep(int resStep)
{
    if(resStep <= 1)
        return 1;
    if(cache.find(resStep) != cache.end())
        return cache[resStep];
    int count = upStep(resStep - 1) + upStep(resStep - 2);
    cache[resStep] = count;
    return count;
}

int main()
{
    int step = 10;
    cout << upStep(step) << endl;
}