#include <iostream>
#include <vector>
#include <algorithm> // 引入algorithm头文件才能使用heap相关操作
using namespace std;

bool compareLess(int num1, int num2)
{
    return num1 > num2;
}

int main()
{
    vector<int> v{6, 1, 2, 5, 3, 4};
    cout << "source vector: ";
    for (auto val : v)
    {
        cout << val << ",";
    }
    cout << endl;
    // make_heap：建堆，默认建立最大堆
    make_heap(v.begin(), v.end());
    cout << "heap vector: ";
    for(auto val: v)
    {
        cout << val << ",";
    }
    cout << endl;

    // push_heap: 当堆中增加一个元素至最末尾时，用来调整堆
    v.push_back(8);
    push_heap(v.begin(), v.end());
    cout << "push_heap:";
    for (auto val : v)
    {
        cout << val << ",";
    }
    cout << endl;

    // pop_heap：将堆顶元素与最后一个叶子节点交换位置，并调整堆（不包括交换位置后的原堆顶元素），此时最后一个节点就是弹出的堆顶元素
    pop_heap(v.begin(), v.end());
    cout << "pop vector: ";
    for (auto val : v)
    {
        cout << val << ",";
    }
    cout << endl;

    // make_heap默认建立最大堆，可以通过辅助函数定义建堆规则，如这里变成建立最小堆
    make_heap(v.begin(), v.end(), compareLess);
    cout << "min heap: ";
    for (auto val : v)
    {
        cout << val << ",";
    }
    cout << endl;

    // sort_heap：堆排序算法，通过每次弹出堆顶元素直至堆为空
    // 排序结束后已经是一个有序列表，不再是堆了
    sort_heap(v.begin(), v.end(), compareLess);
    cout << "sort heap: ";
    for (auto val : v)
    {
        cout << val << ",";
    }
    cout << endl;

    // is_heap：判断是否是堆
    vector<int> v1{6, 1, 2, 5, 3, 4};
    cout << is_heap(v1.begin(), v1.end()) << endl;
    make_heap(v1.begin(), v1.end());
    cout << is_heap(v1.begin(), v1.end()) << endl;

    // is_heap_until：返回第一个不满足heap条件的元素
    vector<int> v2{6, 1, 2, 5, 3, 4};
    auto iter = is_heap_until(v2.begin(), v2.end());
    cout << *iter << endl;

    return 0;
}