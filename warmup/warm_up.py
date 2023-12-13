class Node:
    def __init__(self, data, head):
        self.item = data
        self.next = head
        
class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
        
    def add(self, item):
        self.head = Node(item, self.head)
        self.size += 1

    def remove(self):
        if self.is_empty():
            return None
        else:
            item = self.head.item
            self.head = self.head.next
            return item

    def is_empty(self):
        return self.head is None