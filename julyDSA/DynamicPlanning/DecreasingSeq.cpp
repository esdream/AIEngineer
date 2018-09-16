// 输入一个int序列, 求最长递减序列长度
// 例如输入[18, 16, 15, 19, 20, 11, 10, 9, 8, 7], 返回8
// 有两种解法：
// 第一种是将原序列按从大到小排列，然后求二者的最长公共子序列，复杂度为O(nlogn) + O(n^2) = O(n^2)
#include <iostream>
#include <vector>
#include <map>
using namespace std;

int max(int a, int b)
{
    return a > b ? a : b;
}

void quickSort(vector<int>& arr, int low, int high)
{
    if(low < high)
    {
        int l = low;
        int r = high;
        int key = arr[l];
        while(l < r)
        {
            while(l < r && key >= arr[r])
                --r;
            arr[l] = arr[r];
            while(l < r && key <= arr[l])
                ++l;
            arr[r] = arr[l];
        }
        arr[l] = key;
        quickSort(arr, low, l - 1);
        quickSort(arr, r + 1, high);
    }
}

// 求最长公共子序列
int lcs(vector<int> &a, vector<int> &b, int m, int n)
{
    vector<vector<int>> cache(m + 1, vector<int>(n + 1));
    for (int i = 0; i < m + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            if (i == 0 || j == 0)
                cache[i][j] = 0;
            else if (a[i - 1] == b[j - 1])
                cache[i][j] = cache[i - 1][j - 1] + 1;
            else
                cache[i][j] = max(cache[i - 1][j], cache[i][j - 1]);
        }
    }
    return cache[m][n];
}

int main()
{
    int a[8] = {9, 4, 3, 2, 5, 4, 3, 2};
    vector<int> srcArr(a, a + 8);
    vector<int> sortedArr = srcArr;
    cout << endl;
    cout << lcs(srcArr, sortedArr, srcArr.size(), sortedArr.size()) << endl;
    return 0;
}