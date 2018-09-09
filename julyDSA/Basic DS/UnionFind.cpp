/* 并查集实现
 */
# include <iostream>
using namespace std;

class UnionFind
{
public:
    UnionFind(int size);
    ~UnionFind();
    int find(int x);
    void join(int x, int y);
    int *pre;

private:
    int size_;
};

UnionFind::UnionFind(int size)
{
    size_ = size;
    pre = new int[size_];
    for (int i = 0; i < size_; i++)
        pre[i] = i;
}
UnionFind::~UnionFind()
{
    delete[] pre;
}
int UnionFind::find(int x)
{
    int root = x, tmp;
    // 第一步：向上搜索到根节点
    while(pre[root] != root)
        root = pre[root];
    // 第二步：更新路径上所有节点的父节点为根节点
    while (pre[x] != root)
    {
        tmp = pre[x];
        pre[x] = root;
        x = tmp;
    }
    return root;
}
void UnionFind::join(int x, int y)
{
    int root_x = find(x), root_y = find(y);
    if(root_x != root_y)
        pre[root_x] = root_y;
}

int main()
{
    UnionFind stuUnionFind(10);
    stuUnionFind.join(2, 3);
    stuUnionFind.join(5, 4);
    cout << stuUnionFind.find(2) << endl;
    cout << stuUnionFind.find(5) << endl;
    for (int i = 0; i < 10; i++)
        cout << stuUnionFind.pre[i] << " ";
    return 0;
}