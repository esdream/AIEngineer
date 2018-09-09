#include <iostream>
#include <vector>
#include <stack>
using namespace std;

class Solution
{
  public:
    string removeKdigits(string num, int k)
    {
        if (k == num.length())
        {
            return "0";
        }

        delNum.push(num[0]);
        int idx = 1, delCount = 0;

        while (delCount <= k)
        {
            if (!delNum.empty() && num[idx] < delNum.top())
            {
                delNum.pop();
                delCount += 1;
            }
            else
            {
                delNum.push(num[idx]);
                idx += 1;
            }
        }

        string returnStr = "";
        // 反转字符顺序
        while (!delNum.empty())
        {
            reverseStack.push(delNum.top());
            delNum.pop();
        }

        // 拼接成字符串
        while (!reverseStack.empty())
        {
            returnStr += reverseStack.top();
            reverseStack.pop();
        }

        // 把原来剩的字符串与前面的拼起来
        returnStr += num.substr(idx - 1, num.length());

        int notZeroIdx = 0;
        while (returnStr[notZeroIdx] == '0')
        {
            notZeroIdx++;
        }

        return returnStr.substr(notZeroIdx, returnStr.length());
    }

  private:
    stack<char> delNum;
    stack<char> reverseStack;
};

int main()
{
    Solution s;
    string result = s.removeKdigits("10200", 1);
    cout << result << endl;
    return 0;
}