/* 最长公共子序列 —— 自顶向下递归解法

只能求出最大公共子序列的长度，求不出具体的最大公共子序列
 */

#include <iostream>
#include <algorithm>
#include <string>
using namespace std;

int maxSubSeries(int idx1, int idx2, string X, string Y)
{
    int lenX = X.size(), lenY = Y.size();
    if(idx1 >= lenX || idx2 >= lenY) {
        return 0;
    }
    if(X[idx1] == Y[idx2]) {
        return 1 + maxSubSeries(idx1 + 1, idx2 + 1, X, Y);
    }
    else
        return max(maxSubSeries(idx1 + 1, idx2, X, Y), maxSubSeries(idx1, idx2 + 1, X, Y));
}

int main() {
    string X = "ABCBDAB";
    string Y = "BDCABA";
    cout << maxSubSeries(0, 0, X, Y) << endl;

    // 当使用指针指向vector时，必须使用迭代器iterator才能访问vector中的元素
    // for (vector<char>::iterator it = series->begin(); it != series->end(); it++) {
    //     cout << *it << " ";
    // }

    return 0;
}