function [error] = logistic_xval_error(X, Y, part)
% LOGISTIC_XVAL_ERROR - Logistic regression cross-validation error.
%
% Usage:
%
%   ERROR = logistic_xval_error(X, Y, PART)
%
% Returns the average N-fold cross validation error of the logistic regression
% algorithm on the given dataset when the dataset is partitioned according to PART 
% (see MAKE_XVAL_PARTITION).
%
% Note that N = max(PART).
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, LOGISTIC_REGRESSION


% FILL IN YOUR CODE HERE

iterations=400;
step_size =0.2;
    
N = max(part);
error_temp = zeros(N,1);

for i = 1:N
    indx_train = find(part~=i);
    Xtrain = X(indx_train,:);
    Ytrain = Y(indx_train);
    indx_test = find(part==i);
    Xtest = X(indx_test,:);
    Ytest = Y(indx_test);
    labels = logistic_regression(Xtrain,Ytrain,Xtest,step_size,iterations);
    error_temp(i) = mean(labels ~= Ytest);
end
error = mean(error_temp);