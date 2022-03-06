import numpy as np
import math


class tree_node:
    """
    A node can either be a normal node or a leaf node. A normal node has a connection
    to the left node and the right node, but a leaf node does not have any children.
    """
    
    def __init__(self, center, split, leaf=False, left = None, right=None):
        self.center = center
        self.split = split
        if not leaf:
            self.left = left
            self.right = right
        return

class decisionTree:
    """
    A decision tree takes as input a numpy array of training features and training labels, and
    creates a decision tree model depending on the type of model required and given set of 
    hyper-parameters. 
    """
    def __init__(self, type="classification", max_depth = 10):
        self.type = type
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
        for col in range(ncol):
            comb = np.column_stack((features[:, col], labels))
            comb = comb[np.argsort(comb[:, 0])]
            best_split = None
            best_score = math.inf
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
            
    def divide_data(self, features, labels, col, split):
        leftGroup, rightGroup = [], []
        for i in range(len(features)):
            if features[i, col] <= split:
                leftGroup.append(i)
            else:
                rightGroup.append(i)
        return leftGroup, rightGroup
            

    def train(self, trainFeatures, trainLabels):
        """
        This method trains a decision tree model given then training data. A decision tree 
        is nothing but a tree data structure.
        """
        self.trainFeatures = trainFeatures
        self.trainLabels = trainLabels
        nrows, ncols = trainFeatures.shape
        if self.type == "classification":
            while True:
                best_col, best_split = self.best_split(trainFeatures, trainLabels)
                leftGroup, rightGroup = self.divide_data(trainFeatures, trainLabels, best_col, best_split)
                bleft_col, bleft_split = self.best_split(trainFeatures[leftGroup], trainLabels[leftGroup])
                bright_col, bright_split = self.best_split(trainFeatures[rightGroup], trainLabels[rightGroup])
                tree_node(best_col, best_split, left=bleft_col, right=bright_col)
                    
            