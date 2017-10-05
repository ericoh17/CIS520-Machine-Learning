This folder contains a 2-dimensional synthetic data for binary classification. In the data files provided, each row is a separate training example; the first 2 columns are the features, while the last column has the labels (+1/-1). 

Files:
train.mat - Training instances + labels (1000 x 3)
test.mat - Test instances + labels (1000 x 3)

Folder:
CrossValidation - contains the cross-validation data for each fold of the 5-fold cross-validation procedure on the training set.

The cross-validation data has been generated using the following procedure: we divided the training data into 5 equal parts. For each fold cv_train contains 4 of the 5 parts and cv_test contains 1 left-out part. The parts for training for each fold are different, and also the parts for test for each fold are different.

For doing cross-validation, you take each fold you and train on the cv_train, and the error for the fold would be calculated using cv_test. Select the parameters with the least average cross-validation error.

