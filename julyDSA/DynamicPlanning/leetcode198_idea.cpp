// 解法1：自顶向下：递归 + 缓存的方法
class Solution1
{
public:
    int robot(int idx, vector<int>& nums)
    {
        if(idx >= nums.size())
        {
            return 0;
        }
        
        if(cache.find(idx) != cache.end())
        {
            return cache[idx];
        }

        int a = nums[idx] + robot(idx + 2, nums);
        int b = 0 + robot(idx + 1, nums);

        // return max(a, b);
        // 在原来递归的返回处存缓存
        int maxVal = max(a, b);
        cache[idx] = maxVal;
        return maxVal;
    }
    int max(int a, int b)
    {
        return a > b ? a : b;
    }
    int rob(vector<int> &nums)
    {
        cache.clear();
        if (nums.size() == 0) return 0;
        return robot(0, nums);
    }

private:
    // DP一般使用map加缓存
    map<int, int> cache;
};


// 解法2：自底向上，不使用递归
class Solution2
{
public:
    int rob(vector<int>& nums)
    {
        cache.clear();
        int len = nums.size();
        if(len == 0) return 0;
        // 只剩最后一家时，不论如何都必须要偷（否则偷到的值为0）
        cache[len - 1] = nums[len - 1];
        // 自底向上逐步推导
        for (int idx = len - 2; idx >= 0; --idx)
        {
            // idx + 2是已经解决了的问题
            int a = nums[idx] + (cache.find(idx + 2) != cache.end() ? cache[idx + 2] : 0);
            int b = 0 + (cache.find(idx + 1) != cache.end() ? cache[idx + 1] : 0);
            int maxVal = max(a, b);
            cache[idx] = maxVal;
        }

        return cache[0];
    }

    int max(int a, int b)
    {
        return a > b ? a : b;
    }

private:
    map<int, int> cache;
};


// 解法3：自底向上，并去掉边界判断
class Solution3
{
  public:
    int rob(vector<int> &nums)
    {
        cache.clear();
        int len = nums.size();
        // 添加没有住户可偷时的边界
        if (len == 0)
            return 0;
        // 添加只剩一家时的边界
        if (len == 1)
            return nums[0];
        // 只剩最后一家时，不论如何都必须要偷（否则偷到的值为0）
        cache[len - 1] = nums[len - 1];
        // 只剩两家时，考虑两者最大
        cache[len - 2] = max(nums[len - 1], nums[len - 2]);
        // 自底向上逐步推导
        for (int idx = len - 2; idx >= 0; --idx)
        {
            // idx + 2是已经解决了的问题
            // int a = nums[idx] + cache[idx + 2];
            // int b = cache[idx + 1];
            // int maxVal = max(a, b);
            cache[idx] = max(nums[idx] + cache[idx + 2], cache[idx + 1]);
        }

        return cache[0];
    }

    int max(int a, int b)
    {
        return a > b ? a : b;
    }

private:
    // 这里也可以用数组来替代map，这样比较省空间，但前提是DP中的子问题是按顺序离散的
    map<int, int> cache;
};