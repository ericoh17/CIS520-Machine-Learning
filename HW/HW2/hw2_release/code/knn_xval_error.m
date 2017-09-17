function [error] = knn_xval_error(X, Y, K, part, distFunc)
% KNN_XVAL_ERROR - KNN cross-validation error.
%
% Usage:
%
%   ERROR = knn_xval_error(X, Y, K, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the K-NN algorithm on the 
% given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART), corresponding to the number of folds.
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, K_NEAREST_NEIGHBOURS

N = max(part) ;
error_mat = zeros(N, 1) ;

for i = 1:N 
    
    % get training data for ith fold
    ind_train = find(part ~= i) ;
    Xtrain = X(ind_train, :) ;
    Ytrain = Y(ind_train) ;
    
    % get testing data for ith fold
    ind_test = find(part == i) ;
    Xtest = X(ind_test, :) ;
    Ytest = Y(ind_test) ;
    
    % get predicted labels
    labels = k_nearest_neighbours(Xtrain, Ytrain, Xtest, K, distfunc) ;
    
    % calculate error for ith fold
    error_mat(i) = mean(labels ~= Ytest) ;
    
end
    
error = mean(error_mat) ;
    
    
    
    