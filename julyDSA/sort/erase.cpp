#include <iostream>
#include <vector>
using namespace std;

int main()
{
    int a[3] = {1, 2, 3};
    vector<int> test(a, a+3); 
    for(int i = 0; i < test.size(); i++)
    {
        cout << test[i] << " ";
    }
    cout << endl;

    test.erase(test.begin());
    for (int i = 0; i < test.size(); i++)
    {
        cout << test[i] << " ";
    }

    return 0;
}