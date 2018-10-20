#include <iostream>
using namespace std;
class Solution
{
  public:
    int removeElement(int A[], int n, int elem)
    {
        if (A == NULL || n < 1)
            return n;
        for (int i = 0; i < n; i++)
        {
            if (A[i] == elem)
            {
                while (n > i && A[--n] == elem);
                A[i] = A[n];
            }
        }
        return n;
    }
};

int main()
{
    int a[5] = {2, 4, 6, 4, 3};
    Solution s;
    int length = s.removeElement(a, 5, 4);
    cout << length << endl;
    for(int i = 0; i < 5; i++)
        cout << a[i] << " ";
    return 0;
}