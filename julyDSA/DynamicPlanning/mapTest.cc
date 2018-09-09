#include <iostream>
#include <map>
#include <string>
using namespace std;

void printMap(map<int, string>& m, string pre) {
    map<int, string>::iterator it;
    cout << pre;
    for(it = m.begin(); it != m.end(); it++) {
        cout << "(" << it->first << "," << it->second << ")";
        cout << endl;
    }
}

int main() {
    map<int, string> myMap;
    myMap[1] = "test";
    myMap[2] = "dark";
    myMap[3] = "hyun";
    cout << myMap[1] << endl;

    // 在map中查找
    // 要查找的key
    int nFindKey = 4;
    // 定义一个条目变量（实际是指针）
    map<int, string>::iterator it = myMap.find(nFindKey);
    // 如果it == myMap.end()，表示没有找到
    if(it == myMap.end()) {
        cout << "not found" << endl;
    }
    else {
        cout << myMap[nFindKey] << endl;
    }

    // 插入
    // 单个插入
    myMap.insert(map<int, string>::value_type(5, "insert1"));
    myMap.insert(map<int, string>::value_type(6, "insert2"));
    // 插入一个范围
    map<int, string> myMap2;
    myMap2.insert(myMap.begin(), myMap.end());

    printMap(myMap2, "插入一个范围得到myMap2:\n");

    // 从map中删除元素
    myMap.erase(1);
    printMap(myMap, "删除主键为1的元素后的myMap:\n");
    // 迭代器删除
    map<int, string>::iterator iter = myMap.find(3);
    myMap.erase(iter);
    printMap(myMap, "迭代器删除主键为3的元素后的myMap:\n");

    // 清空
    myMap.clear();
    printMap(myMap, "清空:\n");

    return 0;
}