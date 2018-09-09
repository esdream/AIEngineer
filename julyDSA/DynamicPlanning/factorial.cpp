/* 阶乘动态规划实现：递归 + 缓存
 */
#include <iostream>
#include <map>
using namespace std;

static map<int, int> cache;

int factorial(int n) {
    if(n == 1)
    {
        return 1;
    }
    if(cache.find(n) != cache.end())
        return cache[n];
    cache[n] = n * factorial(n - 1);
    return cache[n];
}

int main() {
    cache.clear();
    int n = 10;
    cout << factorial(n) << endl;
    return 0;
}