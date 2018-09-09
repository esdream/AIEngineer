#include <iostream>
using namespace std;

int tmp = 0;

// 时间复杂度小于O(2^n)，空间复杂度O(n)——栈的深度，说明递归并不省空间
int fib(int n) {
    if(n < 2) {
        return 1;
    }

    else {
        return fib(n - 1) + fib(n - 2); // 由于fib在每次任务拆成两部分，但规模减小，因此时间复杂度小于2^n
    }
}

// 动态规划，递归+缓存，时间复杂度O(n)，空间复杂度O(n)
int fibDP(int n) {
    int *f = new int[n];
    f[0] = 1, f[1] = 1; // 边界条件
    for(int i = 2; i < n; i++) {
        f[i] = f[i-1] + f[i-2];
    }
    return f[n-1];
}

int main() {
    int n = 10;
    for(int i = 1; i < n; i++) {
        cout << fibDP(i) << endl;
    }
    return 0;
}