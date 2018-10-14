/* 二分查找
给定一个单调不降的int数组与int x，求最后一个小于x的元素下标。若不能找到，则返回-1
注意：二分查找过程中，最后一次二分，可能是i向右移动使得不满足i < j条件终止，也可能是j向左移动使得不满足i < j条件终止，因此必须在while循环结束后再做一次判断，如果是i向右移动后终止的，则return i - 1，否则return i
 */
#include <iostream>
#include <vector>
using namespace std;

int binSearch(vector<int>& arr, int x)
{
    int i = 0, j = arr.size() - 1;
    if (arr[j] < x)
        return j;
    if(arr[i] >= x)
        return -1;
    int mid = i + (j - i) / 2;
    while(i < j)
    {
        if(arr[mid] >= x)
            j = mid - 1;
        else
            i = mid + 1;
        mid = i + (j - i) / 2;
    }
    if(arr[i] >= x)
        return i - 1;
    else
        return i;
}

int main()
{
    vector<int> arr = {1, 2, 2, 2, 4, 4, 4, 6};
    int x = 3;
    cout << binSearch(arr, x) << endl;
    return 0;
}