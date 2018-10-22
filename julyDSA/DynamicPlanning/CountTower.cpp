/* 对输入的数塔，要求从顶层走到底层，每一步只能走相邻节点，则经过的节点的数字之和最大是多少？
输入：
第一行是一个整数N，表示塔的高度
接下来N行表示数塔，第i行有i个整数
例如：输入
5
7
3 8
8 1 0
2 7 4 4
4 5 2 6 5
 */
#include <iostream>
#include <vector>
#include <map>
using namespace std;

map<pair<int, int>, int> cache;

int maxSum(int layer, int pos, vector<vector<int>> tower)
{
    if(layer == 0 && pos == 0)
        return tower[layer][pos];
    pair<int, int> loc(layer, pos);
    if (cache.find(loc) != cache.end())
        return cache[loc];
    int maxValue;
    if(pos == layer)
        maxValue = maxSum(layer - 1, pos - 1, tower) + tower[layer][pos];
    else if(pos == 0)
        maxValue = maxSum(layer - 1, pos, tower) + tower[layer][pos];
    else
        maxValue = max(maxSum(layer - 1, pos, tower), maxSum(layer - 1, pos - 1, tower)) + tower[layer][pos];
    cache[loc] = maxValue;
    return maxValue;
}

int main()
{
    int layer = 5;
    vector<vector<int>> tower(5);
    tower[0] = {7};
    tower[1] = {3, 8};
    tower[2] = {8, 1, 0};
    tower[3] = {2, 7, 4, 4};
    tower[4] = {4, 5, 2, 6, 5};

    int maxNum = 0;
    for (int i = 0; i < layer; i++)
    {
        maxNum = max(maxNum, maxSum(layer - 1, i, tower));
    }
    cout << maxNum << endl;
    return 0;
}