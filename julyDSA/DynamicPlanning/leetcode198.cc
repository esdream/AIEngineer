#include <iostream>
#include <map>
#include <vector>
using namespace std;
class Solution
{
  public:
    Solution()
    {
        map<int, int> cache;
    }

    int max(int a, int b)
    {
        return a >= b ? a : b;
    }

    int robot(int idx, vector<int> &nums)
    {
        // 递归边界
        if (idx >= nums.size())
        {
            return 0;
        }
        // 取cache: 一般在边界条件和递归之间，判断完边界条件后就取cache
        if (this->cache.find(idx) != this->cache.end())
        {
            return this->cache[idx];
        }

        // 子问题
        // 状态转移方程
        int a = nums[idx] + robot(idx + 2, nums);
        int b = 0 + robot(idx + 1, nums);

        // 存cache: 一般在返回答案的地方
        int c = max(a, b);
        this->cache[idx] = c;
        return c;
    }

    int rob(vector<int> &nums)
    {
        // 如果程序只运行一次，那么这一行清除缓存的意义是什么？
        this->cache.clear();

        if (nums.size() == 0)
        {
            return 0;
        }
        return robot(0, nums);
    }

  private:
    // 缓存一般使用map
    map<int, int> cache;
};