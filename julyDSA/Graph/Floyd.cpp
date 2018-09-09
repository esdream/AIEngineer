/* Floyd算法：求任意两点之间最短路径
 */
#include <iostream>
using namespace std;

// 参数：邻接矩阵, 定点个数
void floyd(int(*adj)[10], int v)
{
    // inf表示无穷大
    int inf = 9999999;

    // 下标从1开始
    for (int k = 1; k <= v; k++)
        for (int i = 1; i <= v; i++)
            for (int j = 1; j <= v; j++)
            {
                if (adj[i][k] < inf && adj[k][j] < inf && adj[i][j] > adj[i][k] + adj[k][j])
                    adj[i][j] = adj[i][k] + adj[k][j];
            }
}

int main()
{
    // adj为联接矩阵, v表示顶点个数
    int adj[10][10], v = 4;
    // inf表示无穷大
    int inf = 9999999;
    // 初始化，下标从1开始
    for (int i = 1; i <= v; i++)
        for (int j = 1; j <= v; j++)
        {
            if(i == j)
                adj[i][j] = 0;
            else
                adj[i][j] = inf;
        }
    
    // 写入边与权值
    adj[1][2] = 2;
    adj[1][3] = 6;
    adj[1][4] = 4;
    adj[2][3] = 3;
    adj[3][1] = 7;
    adj[3][4] = 1;
    adj[4][1] = 5;
    adj[4][3] = 12;

    floyd(adj, v);

    for (int i = 1; i <= v; i++)
    {
        for (int j = 1; j <= v; j++)
            cout << adj[i][j] << " ";
        cout << endl;
    }
    return 0;
}
