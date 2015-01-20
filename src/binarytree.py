class BinaryTree:

  def __init__(self, val):
    self.val = val
    self.left = None
    self.right = None
    self.parent = None

  def setLeft(self, node):
    self.left = node
    self.left.parent = self

  def setRight(self, node):
    self.right = node
    self.right.parent = self

  def setChildren(self, node1, node2):
    self.setLeft(node1)
    self.setRight(node2)

  def inorder(self):
    leftVals = self.left.inorder() if self.left is not None else []
    rightVals = self.right.inorder() if self.right is not None else[]
    return leftVals + [self.val] + rightVals
