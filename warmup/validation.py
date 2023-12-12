from warm_up import LinkedList

linked_list = LinkedList()
linked_list.add(1)
linked_list.add(3)
linked_list.add(5)
linked_list.remove()
# check the number
assert len(linked_list) == 2
print('Congrats, test passed')