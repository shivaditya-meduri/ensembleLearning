import numpy as np
import math


class tree_node:
    """
    A node can either be a normal node or a leaf node. A normal node has a connection
    to the left node and the right node, but a leaf node does not have any children.
    """
    
    def __init__(self, col=None, split=None, leaf=False, maj_class = None):
        self.col = col
        self.split = split
        self.left = None
        self.right = None
        self.maj_class = maj_class
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
            if features[self.col]<= self.split:
                self.left.traverse(features)
            else:
                self.right.traverse(features)
        else:
            print("Predicted class is : ", self.maj_class)
            print("Leaf reached!!!!!")
            return self.maj_class


class decisionTree:
    """
    A decision tree takes as input a numpy array of training features and training labels, and
    creates a decision tree model depending on the type of model required and given set of 
    hyper-parameters. 
    """
    def __init__(self, type="classification", max_depth = 10):
        self.root = None
        self.type = type
        self.depth = 0
        self.max_depth = max_depth
        return

    def gini_impurity(self, group):
        """
        This method will calculate the gini-impurity of a group of objects
        gini(group) = 1-pr(class1)^2-pr(class2)^2..... for all classes
        """
        n = len(group)
        classes, counts = np.unique(group, return_counts=True)
        p = 0
        for i in range(len(classes)):
            p += (counts[i]/n)**2
        return 1-p

    def find_split(self, features, labels):
        """
        This method will find the best feature that splits the given dataset
        that results in the minimum impurity possible. it returns the feature to
        be split and the value of that feature best for split.
        """
        nrow, ncol = features.shape    
        best_score_col = math.inf
        best_col, best_split_col = None, None
        best_split = None
        best_score = math.inf
        for col in range(ncol):
            comb = np.column_stack((features[:, col], labels))
            comb = comb[np.argsort(comb[:, 0])]
            for split_index in range(nrow):
                s1 = self.gini_impurity(comb[:split_index, 1]) * split_index/nrow
                s2 = self.gini_impurity(comb[split_index:, 1]) * (nrow - split_index)/nrow
                wg = s1 + s2
                if wg < best_score:
                    best_split = comb[split_index, 0]
                    best_score = wg
            if best_score < best_score_col:
                best_score_col = best_score
                best_col = col
                best_split_col = best_split
        return best_col, best_split_col
            
    def divide_data(self, features, col, split):
        leftGroup, rightGroup = [], []
        for i in range(len(features)):
            if features[i, col] <= split:
                leftGroup.append(i)
            else:
                rightGroup.append(i)
        return leftGroup, rightGroup

    def build_tree(self, features, labels):
        root_col, root_split = self.find_split(features, labels)
        root = tree_node(root_col, root_split)
        self.depth += 1
        leftGroup, rightGroup = self.divide_data(features, root_col, root_split)
        if self.gini_impurity(labels[leftGroup]) >= 0 and self.depth<=self.max_depth:
            leftNode = self.build_tree(features[leftGroup], labels[leftGroup])
            root.left_insert(leftNode)
        else:
            root.left_insert(tree_node(leaf=True, maj_class=labels[leftGroup][0]))
        if self.gini_impurity(labels[rightGroup]) >= 0 and self.depth<=self.max_depth:
            rightNode = self.build_tree(features[rightGroup], labels[rightGroup])
            root.right_insert(rightNode)
        else:
            root.right_insert(tree_node(leaf=True, maj_class=labels[leftGroup][0]))
        return root            

    def train(self, trainFeatures, trainLabels):
        """
        This method trains a decision tree model given then training data. A decision tree 
        is nothing but a tree data structure.
        """

        if self.type == "classification":
            self.root = self.build_tree(trainFeatures, trainLabels)

    def predict(self, features):
        """
        This method will predict the label for a set of features using the trained
        Decision tree model.
        """
        pred = self.root.traverse(features)
        return pred

                    
            