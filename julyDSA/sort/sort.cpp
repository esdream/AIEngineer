#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// 冒泡排序：时间复杂度O(n^2)
vector<int> bubbleSort(vector<int>& arr)
{
    int len = arr.size();
    for (int i = 0; i < len - 1; i++)
        for (int j = 0; j < len - 1 - i; j++)
        {
            if(arr[j] > arr[j+1])
            {
                int temp = arr[j+1];
                arr[j+1] = arr[j];
                arr[j] = temp;
            }
        }
    
    return arr;
}

// 选择排序：时间复杂度O(n^2)
vector<int> selectSort(vector<int>& arr)
{
    int len = arr.size();
    int minIndex, temp;
    for (int i = 0; i < len - 1; i++)
    {
        minIndex = i;
        for (int j = i + 1; j < len; j++)
        {
            if (arr[j] < arr[minIndex])
            {
                minIndex = j;
            }
        }
        temp = arr[i];
        arr[i] = arr[minIndex];
        arr[minIndex] = temp;
    }
    return arr;
}

// 插入排序：时间复杂度O(n^2)
vector<int> insertionSort(vector<int>& arr)
{
    int len = arr.size();
    int preIndex, current;
    for (int i = 1; i < len; i++)
    {
        preIndex = i - 1;
        current = arr[i];
        while(preIndex >= 0 && arr[preIndex] > current)
        {
            arr[preIndex + 1] = arr[preIndex];
            preIndex--;
        }
        arr[preIndex + 1] = current;
    }
    return arr;
}

// 希尔排序：平均时间复杂度O(nlogn)
vector<int> shellSort(vector<int> &arr)
{

    int i, j, gap, len = arr.size();
    for (gap = len / 2; gap > 0; gap /= 2)
    {
        for(i = 0; i < gap; i++)
        {
            for (j = i + gap; j < len; j += gap) // 将整个带排序的记录序列分割成若干子序列
            {
                if(arr[j] < arr[j - gap])
                {
                    int temp = arr[j];
                    int k = j - gap;
                    while (k >= 0 && arr[k] > temp) // 对每个子序列进行直接插入排序
                    {
                        arr[k + gap] = arr[k];
                        k -= gap;
                    }
                    arr[k + gap] = temp;
                }
            }
        }
    }
    return arr;
}

// 归并排序：时间复杂度O(nlogn)：时间复杂度始终是O(nlogn)，代价是需要额外的内存空间
// 使用方法：mergeSort(arr)
void merge(vector<int> &arr, vector<int> &left, vector<int> &right);
void mergeSort(vector<int> &arr)
{
    int len = arr.size();
    if(len <= 1)
        return;
    vector<int> left;
    vector<int> right;
    for(int i = 0; i < len; i++)
    {
        if(i < len / 2)
            left.push_back(arr[i]);
        else
            right.push_back(arr[i]);
    }
    mergeSort(left);
    mergeSort(right);
    arr.clear();
    merge(arr, left, right);
}
void merge(vector<int> &arr, vector<int>& left, vector<int>& right)
{
    int len1 = left.size();
    int len2 = right.size();
    int p1 = 0, p2 = 0;
    while (p1 < len1 || p2 < len2)
    {
        if (p1 >= len1)
            arr.push_back(right[p2++]);
        else if(p2 >= len2)
            arr.push_back(left[p1++]);
        else if(left[p1] < right[p2])
            arr.push_back(left[p1++]);
        else
            arr.push_back(right[p2++]);
    }
}

// 快速排序：平均O(nlog(n))，最差O(n^2)
// 使用方法：quickSort(arr, 0, arr.size() - 1)
void quickSort(vector<int> &arr, int low, int high)
{
    if(low < high)
    {
        int l = low;
        int r = high;
        int key = arr[l];

        while(l < r)
        {
            while(l < r && key <= arr[r])
                --r;
            arr[l] = arr[r];  // 此时key位于l位置，更换了key与比key小的元素
            while(l < r && key >= arr[l])
                ++l;
            arr[r] = arr[l]; // 此时key位于r位置，更换了key与比key大的元素
        }
        arr[l] = key; // 此时l = r
        quickSort(arr, low, l-1);
        quickSort(arr, r+1, high);
    }
}

// 堆排序
// 用数组表示堆时，i节点的父节点下标为(i - 1) / 2，其左右子节点下标分别为 2 * i + 1 和 2 * i + 2
// 堆排序实际就是建堆的过程：1. 建立最大堆（或最小堆）；2. 堆调整
void heapify(vector<int> &arr, int i, int len);

// 建立最大堆
// 建立最大堆时，并不需要对每一个节点都建一次堆
// 对所有叶子节点来说，可以认为已经是一个合法的堆（一个叶子节点就是一个堆）
// 一个n个节点构成的堆，叶子节点序号一定从 n / 2 （向下取整） + 1开始
// 所以建堆从 n / 2向下取整开始，一直减小到0即可
void buildMaxHeap(vector<int> &arr)
{
    int len = arr.size();
    for(int i = len / 2; i >= 0; i--)
    {
        heapify(arr, i, len);
    }
}

// 堆调整
void heapify(vector<int> &arr, int i, int len)
{
    int left = 2 * i + 1,
        right = 2 * i + 2,
        largestIdx = i;
    if (left < len && arr[left] > arr[largestIdx])
    {
        largestIdx = left;
    }
    if(right < len && arr[right] > arr[largestIdx])
    {
        largestIdx = right;
    }
    if(largestIdx != i)
    {
        int temp = arr[i];
        arr[i] = arr[largestIdx];
        arr[largestIdx] = temp;
        heapify(arr, largestIdx, len);
    }
}

vector<int> heapSort(vector<int>& arr)
{
    buildMaxHeap(arr);
    int len = arr.size();
    for(int i = arr.size() - 1; i > 0; i--)
    {
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;
        len--;
        heapify(arr, 0, len);
    }
    return arr;
}

// 计数排序：输入的数据必须是确定范围的整数，时间复杂度O(n + k)，k为max - min
vector<int> countSort(vector<int>& arr)
{
    int len = arr.size();
    int max = arr[0], min = arr[0];
    for(int i = 1; i < len; i++)
    {
        if(arr[i] > max)
            max = arr[i];
        if(arr[i] < min)
            min = arr[i];
    }
    vector<int> objArr(len, 0);
    vector<int> range(max - min + 1, 0);
    for (int i = 0; i < len; i++)
    {
        range[arr[i] - min]++;
    }
    int k = 0;
    for (int j = 0; j < range.size(); j++)
    {
        while(range[j] != 0)
        {
            objArr[k] = j + min;
            range[j]--;
            k++;
        }
    }
    return objArr;
}

// 桶排序：时间复杂度O(n)
// 假设有一组长度为N的待排关键字序列K[1....n]。首先将这个序列划分成M个的子区间(桶) 。然后基于某种映射函数 ，将待排序列的关键字k映射到第i个桶中(即桶数组B的下标 i) ，那么该关键字k就作为B[i]中的元素(每个桶B[i]都是一组大小为N/M的序列)。接着对每个桶B[i]中的所有元素进行比较排序(可以使用快排)。然后依次枚举输出B[0]....B[M]中的全部内容即是一个有序序列。
// vector<int> bucketSort(vector<int> &arr)
// {
//     int len = arr.size();
// }

// 基数排序：时间复杂度O(n)
// 基数排序是按照低位先排序，然后收集；再按照高位排序，然后再收集；依次类推，直到最高位。有时候有些属性是有优先级顺序的，先按低优先级排序，再按高优先级排序。最后的次序就是高优先级高的在前，高优先级相同的低优先级高的在前。
vector<int> bucketSort(vector<int> &arr)
{
    int len = arr.size();
    int mod = 10, dev = 1;
    int max = arr[0], min = arr[0];
    for (int i = 1; i < len; i++)
    {
        if (arr[i] > max)
            max = arr[i];
        if (arr[i] < min)
            min = arr[i];
    }

    // 创建基数桶
    vector<vector<int>> counter;

    for(int i = 0; i < max; i++, dev *= 10, mod *= 10)
    {
        for(int k = 0; k < 10; k++)
        {
            vector<int> temp;
            counter.push_back(temp);
        }
        for(int j = 0; j < len; j++)
        {
            int bucket = arr[j] % mod / dev;
            counter[bucket].push_back(arr[j]);
        }
        int pos = 0;
        for (int j = 0; j < counter.size(); j++)
        {
            int value;
            if (!counter[j].empty())
            {
                while (!counter[j].empty())
                {
                    value = counter[j][0];
                    arr[pos++] = value;
                    counter[j].erase(counter[j].begin());
                }
            }
        }
    }
    return arr;
}

int main()
{
    int a[10] = {5, 9, 2, 1, 8, 6, 3, 7, 6, 2};
    vector<int> unsortedArr(a, a+10);
    
    cout << "Unsorted arr: ";
    // 遍历器遍历
    for (vector<int>::const_iterator iter = unsortedArr.begin(); iter != unsortedArr.end(); iter++)
    {
        cout << *iter << " ";
    }

    cout << endl;

    cout << "Sorted arr: ";
    vector<int> sortedArr = bucketSort(unsortedArr);
    // quickSort(unsortedArr, 0, unsortedArr.size() - 1);
    for (vector<int>::const_iterator iter = sortedArr.begin(); iter != sortedArr.end(); iter++)
    {
        cout << *iter << " ";
    }

    // c++ 11 区间遍历
    // for (auto elem : unsortedArr)
    // {
    //     cout << elem << endl;
    // }

    return 0;
}