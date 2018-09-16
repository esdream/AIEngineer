#include <iostream>
#include <vector>
using namespace std;

// 这里可以将vector<int>替换成string
int lcs(vector<int>& a, vector<int>& b, int m, int n)
{
    vector<vector<int>> cache(m + 1, vector<int>(n + 1));
    for (int i = 0; i < m + 1; i++)
    {
        for(int j = 0; j < n + 1; j++)
        {
            if(i == 0 || j == 0)
                cache[i][j] = 0;
            else if(a[i - 1] == b[j - 1])
                cache[i][j] = cache[i - 1][j - 1] + 1;
            else
                cache[i][j] = max(cache[i - 1][j], cache[i][j - 1]);
        }
    }
    return cache[m][n];
}