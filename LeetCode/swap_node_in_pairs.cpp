#include <iostream>
using namespace std;

struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(NULL) {}
};
class Solution {
public:
    ListNode *swapPairs(ListNode *head) {
        if(head == NULL || head->next == NULL)
            return head;

        ListNode L(-1);
        L.next = head; // 头节点前增加一个节点
        ListNode *pre = &L;  // pre为cur的前一个节点
        ListNode *cur = head; // cur为当前节点
        while(cur != NULL && cur->next != NULL)
        {
            ListNode *aft = cur->next; // aft为cur的后一个节点
            cur->next = aft->next; // cur->next指向aft->next
            aft->next = cur; // aft指向当前节点
            pre->next = aft; // pre指向调整后的aft，此时完成了cur和aft节点的互换
            pre = cur; // pre变为cur，虽然只是移动一个节点，但是由于cur和aft互换了，相当于下标加2
            cur = cur -> next; // cur向后移动一个，虽然只是移动一个节点，但是由于cur和aft互换了，相当于下标加2
        }
        return L.next;
    }
};

int main()
{
    ListNode* l1 = new ListNode(1);
    ListNode* l2 = new ListNode(3);
    ListNode* l3 = new ListNode(5);
    ListNode* l4 = new ListNode(6);
    l1->next = l2;
    l2->next = l3;
    l3->next = l4;
    Solution s;
    ListNode* head = l1;
    while (head)
    {
        cout << head->val << " ";
        head = head->next;
    }
    cout << endl;
    head = s.swapPairs(l1);
    while(head)
    {
        cout << head->val << " ";
        head = head->next;
    }
    cout << endl;
    return 0;
}