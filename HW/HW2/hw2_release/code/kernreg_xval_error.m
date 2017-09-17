function [error] = kernreg_xval_error(X, Y, sigma, part)
% KERNREG_XVAL_ERROR - Kernel regression cross-validation error.
%
% Usage:
%
%   ERROR = kernreg_xval_error(X, Y, SIGMA, PART, DISTFUNC)
%
% Returns the average N-fold cross validation error of the kernel regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION). DISTFUNC is the distance functioned 
% to be used.
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, KERNEL_REGRESSION

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
    labels = kernel_regression(Xtrain, Ytrain, Xtest, sigma) ;
    
    % calculate error for ith fold
    error_mat(i) = mean(labels ~= Ytest) ;
    
end
    
error = mean(error_mat) ;

