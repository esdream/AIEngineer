#include <iostream>
#include <vector>
using namespace std;

class Solution {
public:
    bool canJump(vector<int>& nums)
    {
        if(nums.size() == 0)
            return true;
        bool flag = true;
        for(int i = 0; i < nums.size();)
        {
            flag = true;
            for(int step = 1; step <= nums[i]; step++)
            {
                // 新格子规定的可以跳的格子数大于剩余格子数
                if(nums[i + step] > nums[i] - step)
                {
                    i = i + step;
                    flag = false;
                    if(i >= nums.size() - 1)
                        return true;
                    break;
                }
            }
            if(flag == true)
            {
                i = i + nums[i];
            }
            if(i >= nums.size() - 1)
                return true;
            else if(nums[i] == 0)
                return false;   
        }        
    }
};

int main()
{
    Solution s;
    int a[4] = {1, 1, 0, 1};
    vector<int> test(a, a + 4);
    int result = s.canJump(test);
    cout << result << endl;
    return 0;
}