/* 使用广度优先搜索实现 */

class Solution
{
  public:
    int numIslands(vector<vector<char>> &grid)
    {
        int numIslands = 0;
        if (grid.size() == 0)
            return 0;
        int rows = grid.size();
        int cols = grid[0].size();
        queue<pair<int, int>> q;
        vector<pair<int, int>> neighborCoord = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (grid[i][j] == '1')
                {
                    q.push(make_pair(i, j));
                    numIslands++;
                    while (!q.empty())
                    {
                        int x = q.front().first, y = q.front().second;
                        q.pop();
                        grid[i][j] = '0';
                        for (auto coords : neighborCoord)
                        {
                            int newX = x + coords.first, newY = y + coords.second;
                            if ((newX < 0) || (newY < 0) || (newX >= rows) || (newY >= cols) || (grid[newX][newY] == '0'))
                                continue;
                            grid[newX][newY] = '0';
                            q.push(make_pair(newX, newY));
                        }
                    }
                }
            }
        }
        return numIslands;
    }
};