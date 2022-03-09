import numpy as np
import math
from tree_node import tree_node


class decisionTree:
    """
    A decision tree takes as input a numpy array of training features and training labels, and
    creates a decision tree model depending on the type of model required and given set of 
    hyper-parameters. 
    """
    def __init__(self, type="classification", max_depth = 100, min_samples_leaf=10):
        self.root = None
        self.type = type
        self.left_depth, self.right_depth = 0, 0
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
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


    def cart_cost_reg(self, group):
        """
        This function calculates the CART cost function of a set of points split
        using a column and split-value. This function is used to calculate the best_col
        and the best_split to build the decision tree regressor model. Essentially
        the cost function is the variance. 
        """
        n = len(group)
        if n>1:
            var = group.var()
        else:
            var = 0
        return var*n


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
                if self.type == "classification":
                    s1 = self.gini_impurity(comb[:split_index, 1]) * split_index/nrow
                    s2 = self.gini_impurity(comb[split_index:, 1]) * (nrow - split_index)/nrow
                else:
                    s1 = self.cart_cost_reg(comb[:split_index, 1])
                    s2 = self.cart_cost_reg(comb[split_index:, 1])       
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
            if features[i, col] < split:
                leftGroup.append(i)
            else:
                rightGroup.append(i)
        return leftGroup, rightGroup

    def build_tree(self, features, labels):
        root_col, root_split = self.find_split(features, labels)
        root = tree_node(root_col, root_split)
        leftGroup, rightGroup = self.divide_data(features, root_col, root_split)
        if self.type == "classification":
            if self.gini_impurity(labels[leftGroup]) >= 0 and self.left_depth<=self.max_depth and len(leftGroup)>=self.min_samples_leaf:
                self.left_depth += 1
                leftNode = self.build_tree(features[leftGroup], labels[leftGroup])
                root.left_insert(leftNode)
            else:
                root.left_insert(tree_node(leaf=True, classes_info=np.unique(labels[leftGroup], return_counts=True)))
            if self.gini_impurity(labels[rightGroup]) >= 0 and self.right_depth<=self.max_depth and len(rightGroup)>=self.min_samples_leaf:
                self.right_depth += 1
                rightNode = self.build_tree(features[rightGroup], labels[rightGroup])
                root.right_insert(rightNode)
            else:
                root.right_insert(tree_node(leaf=True, classes_info=np.unique(labels[rightGroup], return_counts=True)))
        elif self.type == "regression":
            if self.cart_cost_reg(labels[leftGroup]) >= 0 and self.left_depth<=self.max_depth and len(leftGroup)>=self.min_samples_leaf:
                self.left_depth += 1
                leftNode = self.build_tree(features[leftGroup], labels[leftGroup])
                root.left_insert(leftNode)
            else:
                root.left_insert(tree_node(leaf=True, reg_avg=labels[leftGroup].mean()))
            if self.cart_cost_reg(labels[rightGroup]) >= 0 and self.right_depth<=self.max_depth and len(rightGroup)>=self.min_samples_leaf:
                self.right_depth += 1
                rightNode = self.build_tree(features[rightGroup], labels[rightGroup])
                root.right_insert(rightNode)
            else:
                root.right_insert(tree_node(leaf=True, reg_avg=labels[rightGroup].mean()))
        return root            

    def train(self, trainFeatures, trainLabels):
        """
        This method trains a decision tree model given then training data. A decision tree 
        is nothing but a tree data structure.
        """
        self.root = self.build_tree(trainFeatures, trainLabels)

    def predict(self, features):
        """
        This method will predict the label for a set of features using the trained
        Decision tree model.
        """
        if len(features.shape) == 1:
            pred = self.root.traverse(features)
        elif len(features.shape) == 2:
            pred = []
            for i in range(len(features)):
                pred.append(self.root.traverse(features[i]))
        return pred

                    
            