#include <iostream>
#include <stack>
#include <queue>
#include <vector>
using namespace std;

// 定义点结构
struct Point
{
    int row;
    int col;
    int step; // 步数
    Point(int x, int y)
    {
        this->row = x;
        this->col = y;
    }
    bool operator != (const Point& rhs)
    {
        if(this->row != rhs.row || this->col != rhs.col)
            return true;
        return false;
    }
    bool operator == (const Point& rhs)
    {
        if (this->row == rhs.row && this->col == rhs.col)
            return true;
        return false;
    }
};

// 获取相邻未被访问的节点
// mark是节点标记矩阵，size与maze矩阵相同，用于记录maze中所有节点的访问情况。每走过1步，把走过的点的值加1
Point getAdjNotVisitNode(int **mark, Point point, int m, int n, Point endP)
{
    Point resP(-1, -1);
    if(point.row - 1 >= 0)
    {
        if(mark[point.row - 1][point.col] == 0 || mark[point.row][point.col] + 1 < mark[point.row - 1][point.col]) // 上节点满足条件
        {
            resP.row = point.row - 1;
            resP.col = point.col;
            return resP;
        }
    }
    if (point.row + 1 < m)
    {
        if (mark[point.row + 1][point.col] == 0 || mark[point.row][point.col] + 1 < mark[point.row + 1][point.col]) // 下节点满足条件
        {
            resP.row = point.row + 1;
            resP.col = point.col;
            return resP;
        }
    }
    if (point.col - 1 >= 0)
    {
        if (mark[point.row][point.col - 1] == 0 || mark[point.row][point.col] + 1 < mark[point.row][point.col - 1]) // 左节点满足条件
        {
            resP.row = point.row;
            resP.col = point.col - 1;
            return resP;
        }
    }
    if (point.col + 1 < n)
    {
        if (mark[point.row][point.col + 1] == 0 || mark[point.row][point.col] + 1 < mark[point.row][point.col + 1]) // 右节点满足条件
        {
            resP.row = point.row;
            resP.col = point.col + 1;
            return resP;
        }
    }
    return resP;
}

// 给定二维迷宫，求可行路径
/* 
Parameters:
    maze: 迷宫
    startP: 开始节点
    endP: 结束节点
    pointStack: 栈，存放路径节点
    vecPath: 存放最短路径
*/
void mazePath(void* maze, int m, int n, Point& startP, Point endP, stack<Point>& pointStack, vector<Point>& vecPath)
{
    // 将给定的任意列数的二维数组还原为指针数组，以支持下标操作
    int **maze2d = new int *[m];
    for (int i = 0; i < m; i++)
    {
        maze2d[i] = (int *)maze + i * n;
    }

    // -1表示该点为墙，不能作为路径起点或终点
    if(maze2d[startP.row][startP.col] == -1 || maze2d[endP.row][endP.col] == -1)
        return;

    // 建立各个节点访问标记，表示节点到起点的权值，也记录了起点到当前节点路径的长度
    int **mark = new int *[m];
    for (int i = 0; i < m; i++)
    {
        mark[i] = new int[n];
    }

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            mark[i][j] = *((int *)maze + i * n + j);

    if(startP == endP) // 起点等于终点
    {
        vecPath.push_back(startP);
        return;
    }

    // 增加一个终点的已被访问的前驱节点集
    vector<Point> visitedEndPointPreNodeVec;
    // 将起点入栈
    pointStack.push(startP);
    mark[startP.row][startP.col] = true;

    // 栈不空并且栈顶元素不为结束节点
    while(pointStack.empty() == false)
    {
        Point adjNotVisitedNode = getAdjNotVisitNode(mark, pointStack.top(), m, n, endP);
        if(adjNotVisitedNode.row == -1) // 如果没有符合条件的相邻节点
        {
            pointStack.pop(); // 回溯到上一个节点
            continue;
        }
        if(adjNotVisitedNode == endP) // 以较短的路径找到了终点
        {
            mark[adjNotVisitedNode.row][adjNotVisitedNode.col] = mark[pointStack.top().row][pointStack.top().col] + 1;
            pointStack.push(endP);
            stack<Point> pointStackTemp = pointStack;
            vecPath.clear();
            while(pointStackTemp.empty() == false)
            {
                vecPath.push_back(pointStackTemp.top());
                pointStackTemp.pop();
            }
            pointStack.pop();  // 将终点出栈

            continue;
        }

        // 入栈并设置访问标志为true
        mark[adjNotVisitedNode.row][adjNotVisitedNode.col] = mark[pointStack.top().row][pointStack.top().col] + 1;
        pointStack.push(adjNotVisitedNode);
    }
}

queue<Point> q;
stack<Point> s;
int direction[4][2] = {
    {-1, 0},
    {1, 0},
    {0, -1},
    {0, 1}
};
// 0表示可通行，1表示墙
int maze[5][5] = {
    {0, 0, 1, 1, 0},
    {0, 0, 0, 1, 1},
    {0, 1, 0, 1, 1},
    {0, 0, 0, 1, 1},
    {0, 0, 0, 0, 0}
};

// DFS求第一条可达路径步数
// 注意：不含权重的DFS不一定能找到最短路径。
// 思路：借助栈实现非递归
// 优点：占用内存小，速度快
// 缺点：找到的只是第一条找到的路径，不一定是最短路径。
/* 
Parameters:
    maze: 转化成二维指针数组的迷宫
    m: 行数
    n: 列数
    start: 起点
    end: 终点
 */
int dfs(int **maze, int m, int n, Point start, Point end)
{
    Point t(-1, -1);
    s.push(start);
    maze[start.row][start.col] = 1;
    while(!s.empty())
    {
        start = s.top();
        s.pop();
        if (start == end)
            return start.step;
        for (int i = 0; i < 4; i ++)
        {
            t.row = start.row + direction[i][0];
            t.col = start.col + direction[i][1];
            if (t.row < 0 || t.row >= m || t.col < 0 || t.col >= n || maze[t.row][t.col] == 1)
                continue;
            t.step = start.step + 1; // 当前点的步数+1
            maze[t.row][t.col] = 1;  // 当前点设置为1表示已访问
            s.push(t);
        }
    }
    return -1; // 如果未找到，返回-1
}

// BFS求最短路径步数
// 思路：借助队列实现非递归
// 优点：找到的一定是最短路
// 缺点：要遍历每一层所有节点并存储在队列中，对内存消耗大
/* 
Parameters:
    maze: 转化成二维指针数组的迷宫
    m: 行数
    n: 列数
    start: 起点
    end: 终点
 */
int bfs(int **maze, int m, int n, Point start, Point end)
{
    Point t(-1, -1);
    q.push(start); // 起点入队
    maze[start.row][start.col] = 1;
    while(!q.empty())
    {
        start = q.front();
        q.pop();
        if(start == end)
            return start.step;
        for (int i = 0; i < 4; i++)
        {
            t.row = start.row + direction[i][0];
            t.col = start.col + direction[i][1];
            if(t.row < 0 || t.row >= m || t.col < 0 || t.col >= n || maze[t.row][t.col] == 1)
                continue;
            t.step = start.step + 1;  // 当前点的步数+1
            maze[t.row][t.col] = 1;  // 当前点设置为1表示已访问
            q.push(t);
        }
    }
    return -1; // 如果未找到，返回-1
}

int main()
{
    int m = 5, n = 5;
    Point start(0, 1), end(0, 4);
    /* 将二维数组转化为指针数组用于向函数传参的操作！！！
    如下：
        1. 创建一个二维指针数组，new出m个指针
        2. m个指针分别指向原数组第i行的起始位置（也就是编译器中寻址的原理）
    完成后即可在函数中使用 **二维指针数组名 作为形参，将指针数组名作为实参，且可在函数中直接使用下标操作原数组中的元素。
     */
    int **maze2d = new int *[m];
    for (int i = 0; i < m; i++)
    {
        maze2d[i] = (int *)maze + i * n;
    }

    cout << dfs(maze2d, 5, 5, start, end) << endl;
    return 0;
}