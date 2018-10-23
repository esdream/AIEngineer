/* 判断一个单链表是否有环，并返回环的第一个节点，空间复杂度O(1)，时间复杂度O(n)
思路：
使用两个指针，一个1次走1步，一个1次走2步。
若两个指针相遇，则存在环。两个指针相遇时从head开始再建一个指针，1次走1步，当与slow指针相遇时即为环的第一个节点。
若快指针fast走到结尾（fast->next == NULL）都没有和slow相遇，则无环，返回NULL。
 */
#include <iostream>
using namespace std;

struct ListNode
{
    ListNode* next;
    int val;
    ListNode(int v): val(v), next(NULL) {}
};

class Solution
{
public:
    ListNode* detectCycle(ListNode* head)
    {
        ListNode* fast = head;
        ListNode* slow = head;
        while(fast && fast->next)
        {
            fast = fast->next->next;
            slow = slow->next;
            if(fast == slow)
                break;
        }
        if(!fast || !fast->next)
            return NULL;
        ListNode* p = head;
        while(p != slow)
        {
            p = p->next;
            slow = slow->next;
        }
        return p;
    }
};