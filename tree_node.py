import numpy as np

class tree_node:
    """
    A node can either be a normal node or a leaf node. A normal node has a connection
    to the left node and the right node, but a leaf node does not have any children.
    """
    
    def __init__(self, col=None, split=None, leaf=False, classes_info = None, reg_avg=None):
        self.col = col
        self.split = split
        self.left = None
        self.right = None
        self.reg_avg = reg_avg
        if classes_info != None:
            self.classes, self.classes_count = classes_info
        else:
            self.classes, self.classes_count = None, None
        self.leaf = leaf
        return
    def left_insert(self, node, leaf=False):
        self.left = node
        return
    def right_insert(self, node, leaf=False):
        self.right = node
        return
    def traverse(self, features):
        if self.leaf==False:
            if features[self.col]< self.split:
                return self.left.traverse(features)
            else:
                return self.right.traverse(features)
        else:
            if self.reg_avg == None:
                if len(self.classes_count) != 0:
                    pred_class = self.classes[np.argmax(self.classes_count)]
                else:
                    pred_class = "None"
                #pred_prob = self.classes_count.max()/self.classes_count.sum()
                #print("Predicted class is : ", pred_class, " with a probability of : ", pred_prob)
                #print("Leaf reached!!!!!")
                return pred_class
            else:
                return self.reg_avg