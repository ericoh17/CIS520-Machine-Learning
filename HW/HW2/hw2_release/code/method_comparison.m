% script for HW2, Question 4. Copmarison of KNN, Kernel Regression, Decision Tree, and Logistic Regression methods

% Loading the data: this loads X, and Ytrain.
load('../data/X_noisy.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('../data/Y.mat'); 

% if using noisy dataset
X = X_noisy ;

% split data into 450 training and 150 testing points
N_total = size(X,1) ;

tf = false(N_total, 1) ;
tf(1:450) = true ;
tf = tf(randperm(N_total)) ;

Xtrain = X(tf, :) ;
Xtest = X(~tf, :) ;
Ytrain = Y(tf) ;
Ytest = Y(~tf) ;

N = size(Xtrain, 1) ; 
distfunc = 'l2' ; 

% set method parameters
fold = 10 ;
K = 3 ; % change to 3 for noisy data
sigma = 9 ; % change to 9 for noisy data

% get partitions for cross-validation
part = make_xval_partition(N, fold) ;

% get training error
errors_xval_knn = knn_xval_error(Xtrain, Ytrain, K, part, distfunc) ;
errors_xval_kern = kernreg_xval_error(Xtrain, Ytrain, sigma, part) ;
% add decision tree 
% add logistic regression

% get testing error
knn_test_pred = k_nearest_neighbours(Xtrain, Ytrain, Xtest, K, distfunc) ;
kern_test_pred = kernel_regression(Xtrain, Ytrain, Xtest, sigma) ;
% add decision tree
% add logistic regression

errors_test_knn = mean(knn_test_pred ~= Ytest) ;
errors_test_kern = mean(kern_test_pred ~= Ytest) ;
% add decision tree
% add logistic regression 







