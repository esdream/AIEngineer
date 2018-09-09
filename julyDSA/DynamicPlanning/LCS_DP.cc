/* 最长公共子序列——自底向上动态规划解法
 */
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
using namespace std;

// 最长公共子序列长度记录矩阵
int series[100][100];

void LCS(string X, string Y, int** condition) {
    int lenX = X.size(), lenY = Y.size();
    for(int i = 0; i < lenX; i++) {
        series[i][0] = 0;
        condition[i][0] = 1;
    }
    for (int j = 0; j < lenY; j++)
    {
        series[0][j] = 0;
        condition[0][j] = 1;
    }
    for(int i = 1; i <= lenX; i++) {
        for(int j = 1; j <= lenY; j++) {
            if(X[i] == Y[j]) {
                series[i][j] = series[i - 1][j - 1] + 1;
                condition[i][j] = 2;
            } else if(X[i] >= Y[j]) {
                series[i][j] = series[i - 1][j];
                condition[i][j] = 3;
            } else {
                series[i][j] = series[i][j-1];
                condition[i][j] = 4;
            }
        }
    }
}

//输出最长公共子序列
void outputLCS(string X, int** condition, int m, int n)
{
    if (m == 0 || n == 0)
    { //要从后往前找,某一个序列找到头了，就结束
        return;
    }
    if (condition[m][n] == 1)
    {
        return;
    }
    else if (condition[m][n] == 2)
    {
        outputLCS(X, condition, m - 1, n - 1);
        cout << X[m - 1] << "  ";
    }
    else if (condition[m][n] == 3)
    {
        outputLCS(X, condition, m - 1, n);
    }
    else
    {
        outputLCS(X, condition, m, n - 1);
    }
}

int main()
{
    string X = "ABCBDAB";
    string Y = "BDCABA";
    // 路径状态记录矩阵
    int **condition = new int *[100];
    int t;
    for (t = 0; t < 100; t++)
    {
        condition[t] = new int[100];
    }
    LCS(X, Y, condition);
    cout << series[X.size()][Y.size()] << endl;
    outputLCS(X, condition, X.size(), Y.size());
    return 0;
}