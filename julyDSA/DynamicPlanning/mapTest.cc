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

    // ��map�в���
    // Ҫ���ҵ�key
    int nFindKey = 4;
    // ����һ����Ŀ������ʵ����ָ�룩
    map<int, string>::iterator it = myMap.find(nFindKey);
    // ���it == myMap.end()����ʾû���ҵ�
    if(it == myMap.end()) {
        cout << "not found" << endl;
    }
    else {
        cout << myMap[nFindKey] << endl;
    }

    // ����
    // ��������
    myMap.insert(map<int, string>::value_type(5, "insert1"));
    myMap.insert(map<int, string>::value_type(6, "insert2"));
    // ����һ����Χ
    map<int, string> myMap2;
    myMap2.insert(myMap.begin(), myMap.end());

    printMap(myMap2, "����һ����Χ�õ�myMap2:\n");

    // ��map��ɾ��Ԫ��
    myMap.erase(1);
    printMap(myMap, "ɾ������Ϊ1��Ԫ�غ��myMap:\n");
    // ������ɾ��
    map<int, string>::iterator iter = myMap.find(3);
    myMap.erase(iter);
    printMap(myMap, "������ɾ������Ϊ3��Ԫ�غ��myMap:\n");

    // ���
    myMap.clear();
    printMap(myMap, "���:\n");

    return 0;
}