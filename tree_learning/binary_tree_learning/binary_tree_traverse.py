class Node:
    def __init__(self,value = None, left = None, right = None):
        self.value = value
        self.left = left
        self.right = right


def pre_traversal(root):
    if root == None:
        return
    print(root.value)
    pre_traversal(root.left)
    pre_traversal(root.right)

def mid_traversal(root):
    if root == None:
        return
    mid_traversal(root.left)
    print(root.value)
    mid_traversal(root.right)

def back_traversal(root):
    if root == None:
        return
    back_traversal(root.left)
    back_traversal(root.right)
    print(root.value)



root_a = Node('A',Node('B',right=Node('C', Node('D'))), Node('E',right=Node('F',Node('G',Node('H'), Node('K')))))

pre_traversal(root_a)
print('next: \n')
mid_traversal(root_a)
print('next: \n')
back_traversal(root_a)
