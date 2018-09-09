#include <iostream>
#include <map>
using namespace std;

static int v[3] = {2, 4, 3};  // 物品价值数组
static int w[3] = {30, 45, 15};  // 物品重量数组
static int n = 3;  // 物品的数量
static map<pair<int, int>, int> cache;

int max(int a, int b)
{
    return a > b ? a : b;
}

// 自顶向下
// 参数：
//    idx: 从哪个开始偷
//    restW: 剩余的背包空间
// 时间复杂度和空间复杂度都为O(n * W)
int robot(int idx, int restW)
{
    if (idx >= n || restW <= 0)
    {
        return 0;
    }
    pair<int, int> state(idx, restW);
    // 取cache
    if (cache.find(state) != cache.end())
        return cache[state];

    int value;
    // !!!注意：01背包问题的特殊性，这里如果restW - w[idx] < 0，那就表示当前物体是不能偷的！需要直接将这个分支去掉
    if(restW - w[idx] < 0)
        value = robot(idx + 1, restW);
    else
        value = max(robot(idx + 1, restW - w[idx]) + v[idx], robot(idx + 1, restW));
    // 存cache
    cache[state] = value;
    return value;
}


// 自底向上 : error!!!自底向上的算法感觉是有问题的，因为restW是不连续的，但这里按照连续的方法做循环，建议采用自顶向下的方法
int check(int idx, int restW)
{
    if(idx >= n || restW <= 0)
        return 0;
    pair<int, int> state(idx, restW);
    return cache[state];
}
int robot2(int idx, int restW)
{
    if (idx >= n || restW <= 0)
    {
        return 0;
    }
    for (int i = n - 1; i >= 0; --i)
    {
        int ww = 0;
        while(ww <= restW)
        {
            pair<int, int> state(i, ww);
            if (restW - w[idx] < 0)
                cache[state] = check(idx + 1, restW);
            else
            {
                if (check(i + 1, restW - w[idx]) + v[idx] > check(idx + 1, restW))
                {
                    cache[state] = check(i + 1, restW - w[idx]) + v[idx];
                    ww += restW;
                }
                else
                {
                    cache[state] = check(idx + 1, restW);
                }
            }            
        }
        // for (int ww = 0; ww <= restW; ++ww)
        // {
        //     pair<int, int> state(i, ww);
        //     if (restW - w[idx] < 0)
        //         cache[state] = check(idx + 1, restW);
        //     else
        //         cache[state] = max(check(i + 1, restW - w[idx]) + v[idx], check(idx + 1, restW));
        // }
    }
    pair<int, int> state(idx, restW);
    return cache[state];
}

// 自底向上 : 滚动数组，优化空间复杂度
int check3(int idx, int restW)
{
    if (idx >= n || restW - w[idx] <= 0)
        return 0;
    // 这里由于max只考虑i和i + 1两种情况，那么考虑完i后，存储i的空间可以直接给i + 2使用（模2）。然后考虑i + 1和i + 2即可。空间复杂度由O(n * W)变为O(W) （2为常数舍去）
    pair<int, int> state(idx % 2, restW);
    return cache[state];
}
int robot3(int idx, int restW)
{
    if (idx >= n || restW - w[idx] <= 0)
    {
        return 0;
    }
    for (int i = n - 1; i >= 0; --i)
    {
        for (int ww = 0; ww <= restW; ++ww)
        {
            pair<int, int> state(i % 2, ww);
            if (restW - w[idx] < 0)
                cache[state] = check3(idx + 1, restW);
            else
                cache[state] = max(check3(i + 1, restW - w[idx]) + v[idx], check3(idx + 1, restW));
        }
    }
    pair<int, int> state(idx, restW);
    return cache[state];
}

int main()
{
    cache.clear();
    cout << robot2(0, 50) << endl;
    return 0;
}