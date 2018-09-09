#include <iostream>
using namespace std;

const int INF = 999999;

// 参数：邻接矩阵, 距离数组, 顶点id, 顶点个数n
void dijkstra(int (*adj)[10], int *dist, int *book, int n)
{
    int u, v;
    for (int i = 1; i <= n - 1; i++)
    {
        int min = INF;
        for (int j = 1; j <= n; j++)
        {
            if(book[j] == 0 && dist[j] < min)
            {
                min = dist[j];
                u = j;
            }
        }
        book[u] = 1;
        for (v = 1; v <= n; v++)
        {
            if(adj[u][v] < INF)
            {
                if(dist[v] > dist[u] + adj[u][v])
                    dist[v] = dist[u] + adj[u][v];
            }
        }
    }
}

int main()
{
    // adj为联接矩阵, v表示顶点个数, v0表示起点下标，dist为距离数组，book为已经确定距离的顶点数组（1为已经确定，0为未确定）
    int adj[10][10], v = 4, v0 = 2, dist[10], book[10];
    // 初始化，下标从1开始
    for (int i = 1; i <= v; i++)
        for (int j = 1; j <= v; j++)
        {
            if (i == j)
                adj[i][j] = 0;
            else
                adj[i][j] = INF;
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
    // 初始化dist数组，初始化时为邻接矩阵中起点所在那一行
    for (int i = 1; i <= v; i++)
        dist[i] = adj[v0][i];
    for (int i = 1; i <= v; i++)
        book[i] = 0;
    // 将book中起点对应元素标为1  
    book[v0] = 1;

    dijkstra(adj, dist, book, v);
    for (int i = 1; i <= v; i++)
        cout << dist[i] << " ";

    return 0;
}