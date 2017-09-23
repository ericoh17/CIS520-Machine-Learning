function [error] = dt_xval_error(X, Y, depth_limit, part) 
% DT_XVAL_ERROR = decision tree cross-validation error
%
% Usage: 
%
%    Error = dt_xval_error(X, Y, depth_limit, part) ;
%
% Returns the average N-fold cross validation error of the decision tree 
% algorithm on the given dataset when the dataset is partitioned 
% according to PART (see MAKE_XVAL_PARTITION).
%
% Note that N = max(PART), corresponding to the number of folds.
%
% SEE ALSO
%   MAKE_XVAL_PARTITION, DT_TRAIN, DT_VALUE

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
    
    num_test = size(Xtest, 1) ; 
    
    % get predicted labels
    train_dt = dt_train(Xtrain, Ytrain, 5) ; 
    dt_test_pred = zeros(num_test, 1) ; 

    for test = 1:num_test 
        test_point = Xtest(test,:) ; 
        pred = dt_value(train_dt, test_point) ; 
        
        if pred >= 0.5
            dt_test_pred(test) = 1 ;
        else 
            dt_test_pred(test) = 0 ;
        end
    end
    
    % calculate error for ith fold
    error_mat(i) = mean(dt_test_pred ~= Ytest) ;
    
end
    
error = mean(error_mat) ;

end
