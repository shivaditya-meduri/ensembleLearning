# Ensemble Learning
An implementation of Decision tree and Random Forest models for classification and regression tasks from scratch. A few sample datasets like Iris, Breast cancer and Boston Housing dataset are also included in the repo for testing purposes. 
![image](https://user-images.githubusercontent.com/77324692/157775726-de4e956d-00d3-469c-aa2c-6befb6607cb7.png)
Decision tree divides the feature space into hyper rectangles, which minimize the impurity/variance of the labels in the resulting hyper rectangles. In this implementation, this mechanism is replicated using the binary tree data structure, and the regions/hyper-rectangles are represented by leaf nodes in the tree. 
For classification, the majority label of samples in the leaf node is chosen as the final prediction and for regression, the average of the labels of the samples in the leaf node is chosen as the final prediction.
![image](https://user-images.githubusercontent.com/77324692/157775761-fa02a9ad-3bd1-4399-85ec-ebe91ff1e735.png)
Random Forest is an ensemble of Decision Tree classifiers, which are trained following a bagging method. After training, the predictions of the base estimators are combined to generate the final robust prediction. In case of classification, a majority vote is chosen as the final prediction and in case of regression, an average is taken of the predictions of the base estimators.
To learn how to implement the models created in this repo, check out the tutorial ipython notebook, copy the notebook to google colab or local jupyter notebook and follow the cloning cells to load the files into runtime environment.
Results of testing various datasets:
![image](https://user-images.githubusercontent.com/77324692/157776016-ac08becd-3df2-4f26-92c7-bd169ce1ff13.png)


