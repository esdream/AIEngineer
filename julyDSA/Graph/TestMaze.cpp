#include <iostream>
#include <stack>
#include <vector>
using namespace std;

struct Point{  
    //行与列
    int row;  
    int col;  
    Point(int x,int y){
        this->row=x;
        this->col=y;
    }

    bool operator!=(const Point& rhs){
        if(this->row!=rhs.row||this->col!=rhs.col)
            return true;
        return false;
    }

    bool operator==(const Point& rhs) const{
        if(this->row==rhs.row&&this->col==rhs.col)
            return true;
        return false;
    }
};  

int maze[5][5]={
    {0, 0, 0, 0,0},
    {0,-1, 0,-1,0},
    {0,-1,-1, 0,0},
    {0,-1,-1, 0,-1},
    {0, 0, 0, 0, 0}
};

//func:获取相邻未被访问的节点
//para:mark:结点标记；point：结点；m：行；n：列;endP:终点
//ret:邻接未被访问的结点
Point getAdjacentNotVisitedNode(int** mark,Point point,int m,int n,Point endP){
    Point resP(-1,-1);
    if(point.row-1>=0){
        if(mark[point.row-1][point.col]==0||mark[point.row][point.col]+1<mark[point.row-1][point.col]){//上节点满足条件
            resP.row=point.row-1;
            resP.col=point.col;
            return resP;
        }
    }
    if(point.col+1<n){
        if(mark[point.row][point.col+1]==0||mark[point.row][point.col]+1<mark[point.row][point.col+1]){//右节点满足条件
            resP.row=point.row;
            resP.col=point.col+1;
            return resP;
        }
    }
    if(point.row+1<m){
        if(mark[point.row+1][point.col]==0||mark[point.row][point.col]+1<mark[point.row+1][point.col]){//下节点满足条件
            resP.row=point.row+1;
            resP.col=point.col;
            return resP;
        }
    }
    if(point.col-1>=0){
        if(mark[point.row][point.col-1]==0||mark[point.row][point.col]+1<mark[point.row][point.col-1]){//左节点满足条件
            resP.row=point.row;
            resP.col=point.col-1;
            return resP;
        }
    }
    return resP;
}

//func：给定二维迷宫，求可行路径
//para:maze：迷宫；m：行；n：列；startP：开始结点 endP：结束结点； pointStack：栈，存放路径结点;vecPath:存放最短路径
//ret:无
void mazePath(void* maze,int m,int n, Point& startP, Point endP,stack<Point>& pointStack,vector<Point>& vecPath){
    //将给定的任意列数的二维数组还原为指针数组，以支持下标操作
    int** maze2d=new int*[m];
    for(int i=0;i<m;++i){
        maze2d[i]=(int*)maze+i*n;
    }

    if(maze2d[startP.row][startP.col]==-1||maze2d[endP.row][endP.col]==-1)
        return ;                    //输入错误

    //建立各个节点访问标记，表示结点到到起点的权值，也记录了起点到当前结点路径的长度
    int** mark=new int*[m];
    for(int i=0;i<m;++i){
        mark[i]=new int[n];
    }
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            mark[i][j]=*((int*)maze+i*n+j);
        }
    }
    if(startP==endP){//起点等于终点
        vecPath.push_back(startP);
        return;
    }

    //增加一个终点的已被访问的前驱结点集
    vector<Point> visitedEndPointPreNodeVec;

    //将起点入栈
    pointStack.push(startP);
    mark[startP.row][startP.col]=true;

    //栈不空并且栈顶元素不为结束节点
    while(pointStack.empty()==false){
        Point adjacentNotVisitedNode=getAdjacentNotVisitedNode(mark,pointStack.top(),m,n,endP);
        if(adjacentNotVisitedNode.row==-1){ //没有符合条件的相邻节点
            pointStack.pop(); //回溯到上一个节点
            continue;
        }
        if(adjacentNotVisitedNode==endP){//以较短的路劲，找到了终点,
            mark[adjacentNotVisitedNode.row][adjacentNotVisitedNode.col]=mark[pointStack.top().row][pointStack.top().col]+1;
            pointStack.push(endP);
            stack<Point> pointStackTemp=pointStack;
            vecPath.clear();
            while (pointStackTemp.empty()==false){
                vecPath.push_back(pointStackTemp.top());//这里vecPath存放的是逆序路径
                pointStackTemp.pop();
            }
            pointStack.pop(); //将终点出栈

            continue;
        }
        //入栈并设置访问标志为true
        mark[adjacentNotVisitedNode.row][adjacentNotVisitedNode.col]=mark[pointStack.top().row][pointStack.top().col]+1;
        pointStack.push(adjacentNotVisitedNode);
    }
}

int main(){
    Point startP(0,0);
    Point endP(4,4);
    stack<Point>  pointStack;
    vector<Point> vecPath;
    mazePath(maze,5,5,startP,endP,pointStack,vecPath);

    if(vecPath.empty()==true)
        cout<<"no right path"<<endl;
    else{
        cout<<"shortest path:";
        for(auto i=vecPath.rbegin();i!=vecPath.rend();++i)
            printf("(%d,%d) ",i->row,i->col);
    }

    getchar();
}