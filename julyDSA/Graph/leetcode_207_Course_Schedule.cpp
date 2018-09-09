/* 使用邻接表结构实现
 */

#include <iostream>
#include <vector>
#include <map>
#include <queue>
using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        if (prerequisites.empty())
            return true;

        v_ = numCourses;
        int edgesNum = prerequisites.size();
        // 初始化邻接表
        adjTable_ = vector<vector<int>>(numCourses);
        for (int i = 0; i < numCourses; i++)
        {
            vertexInDeg_[i] = 0;
        }
        for (int j = 0; j < edgesNum; j++)
        {
            addEdge(prerequisites[j]);
        }

        for(int i = 0; i < numCourses; i++)
            if (vertexInDeg_[i] == 0)
                zeroDeg_.push(i);
        
        int count = 0;
        while(!zeroDeg_.empty())
        {
            int zeroVertex = zeroDeg_.front();
            zeroDeg_.pop();
            ++count;
            for (int w = 0; w < adjTable_[zeroVertex].size(); w++)
            {
                if ((--(vertexInDeg_[adjTable_[zeroVertex][w]])) == 0)
                {
                    zeroDeg_.push(adjTable_[zeroVertex][w]);
                }
            }
        }
        if(count < numCourses)
            return false;
        else
            return true;

    }
    
    void addEdge(pair<int, int>& prerequisty)
    {
        adjTable_[prerequisty.second].push_back(prerequisty.first);
        vertexInDeg_[prerequisty.first] += 1;
    }

  private:
    vector<vector<int>> adjTable_;
    map<int, int> vertexInDeg_;
    queue<int> zeroDeg_;
    int v_;
};

int main()
{
    Solution s;
    vector<pair<int, int>> edges;
    edges.push_back(std::make_pair(1, 0));
    edges.push_back(std::make_pair(2, 1));
    bool isFinish = s.canFinish(3, edges);
    cout << isFinish << endl;
    return 0;
}