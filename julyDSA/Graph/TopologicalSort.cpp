/* 使用邻接矩阵结构实现
 */

#include <iostream>
#include <vector>
#include <map>
#include <queue>
using namespace std;

class Graph
{
private:
    int v_;  // 顶点个数
    vector<vector<int>> adjMat_;  // 邻接矩阵
    map<int, int> vertexInDeg_;  // 维护一个节点度hash map
    queue<int> zeroDeg_;  // 度为0的队列

public:
    Graph(int v); // 构造函数
    void addEdge(int v1, int v2);  // 添加边
    bool topologicalSort();  // 拓扑排序
    friend ostream &operator<<(ostream &out, Graph &G);
};

Graph::Graph(int v)
{
    v_ = v;
    adjMat_ = vector<vector<int>>(v);
    for (int i = 0; i < v; i++)
    {
        adjMat_.push_back(vector<int>(v));
        for (int j = 0; j < v; j++)
        {
            adjMat_[i].push_back(0);
        }
        vertexInDeg_[i] = 0;
    }
}

void Graph::addEdge(int v1, int v2)
{
    adjMat_[v1][v2] = 1;
    vertexInDeg_[v2] += 1;
}

bool Graph::topologicalSort()
{
    cout << "Topological sort reuslt: " << endl;
    for (int i = 0; i < v_; i++)
        if (vertexInDeg_[i] == 0)
            zeroDeg_.push(i);

    int count = 0;
    while(!zeroDeg_.empty())
    {
        int zeroVertex = zeroDeg_.front();
        zeroDeg_.pop();
        cout << zeroVertex << " ";
        ++count;
        for (int j = 0; j < v_; j++)
        {
            if (adjMat_[zeroVertex][j] == 1 && (--vertexInDeg_[j]) == 0)
            {
                zeroDeg_.push(j);
            }
        }
    }
    if(count < v_)
        return false;
    else
        return true;
}

ostream & operator << (ostream & out, Graph & G)
{
    for (int i = 0; i < G.v_; i++)
    {
        for (int j = 0; j < G.v_; j++)
            out << G.adjMat_[i][j] << " ";
        out << endl;
    }
    return out;
}

int main()
{
    Graph classes(6);
    classes.addEdge(5, 2);
    classes.addEdge(5, 0);
    classes.addEdge(4, 0);
    classes.addEdge(4, 1);
    classes.addEdge(2, 3);
    classes.addEdge(3, 1);

    cout << classes;

    classes.topologicalSort();
    return 0;
}