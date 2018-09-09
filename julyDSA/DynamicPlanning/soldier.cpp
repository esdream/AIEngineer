#include <iostream>
#include <map>
using namespace std;

static map<pair<int, int>, int> cache1;

// 自顶向下的方式
// 形参括号里一定是区别的条件
int stepCountTopToBottom(int N, int M)
{
    // 边界条件，当往上走或往右走时，只有1种走法
    if(N == 1 || M == 1)
        return 1;
    pair<int, int> loc(N, M);
    if(cache.find(loc) != cache.end())
        return cache[loc];
    int count = stepCountTopToBottom(N, M - 1) + stepCountTopToBottom(N - 1, M);
    // 存cache
    cache1[loc] = count;
    // 取cache
    return cache1[loc];
}

// 自底向上的方式
int cache2[100][100];

// 边界条件：第一行和第一列全部设置为1，表示当只走1行或者1列时，不论走了多少格，都只有一种走法
int stepCountBottomToTop(int N, int M) {
    for (int i = 1; i <= N; i++)
    {
        cache2[i][1] = 1;
    }
    for (int j = 1; j <= M; j++)
    {
        cache2[1][j] = 1;
    }
    for (int x = 2; x <= N; x++)
    {
        for (int y = 2; y <= M; y++)
        {
            cache2[x][y] = cache2[x - 1][y] + cache2[x][y - 1];
        }
    }
    return cache2[N][M];
}

int main() {
    int N = 4, M = 4;
    cout << stepCountTopToBottom(N, M) << endl;
    return 0;
}