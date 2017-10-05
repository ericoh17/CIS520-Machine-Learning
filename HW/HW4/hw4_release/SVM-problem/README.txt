In this directory, the following files are provided.

 1. SVM_train.m  -  (Template) This function is used to learn a SVM model for the specified kernel. You should cross-validate for the kernel parameter, and also C. The train data and the cross-validation data should be loaded from the specified folders. You should specify the test error values you got for both the kernels poly and RBF.

Important: Please do not change directory structure for the autograder to work.
Use libsvm-3.22 (download it from the libsvm website) for training the SVMs for each of the parameters.


 2. decision_boundary_SVM.m - This function plots the decision boundary given the data points with respect to the SVM classifier learnt using SVM_learner.m
 3. classification_error.m  - This functions computes the classification error given the predicted and true labels.
 


The plots would go in the Plots folder.
