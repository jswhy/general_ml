class Node:
    def __init__(self,value = None, left = None, right = None):
        self.value = value
        self.left = left
        self.right = right


def preTraverse(root):
    if root is None:
        return
    print(root.value)
    preTraverse(root.left)
    preTraverse(root.right)

def midTraverse(root):
    if root is None:
        return
    midTraverse(root.left)
    print(root.value)
    midTraverse(root.right)


def backTraverse(root):
    if root == None:
        return
    backTraverse(root.left)
    backTraverse(root.right)
    print(root.value)


node = Node('A',left= Node('B'), right = Node('C'))
preTraverse(backTraverse(node))