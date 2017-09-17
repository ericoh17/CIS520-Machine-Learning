function labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K,distfunc)

    % Function to implement the K nearest neighbours algorithm on the given
    % dataset.
    % Usage: labels = k_nearest_neighbours(Xtrain,Ytrain,Xtest,K)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % K : number of nearest neighbours used to make predictions on the test
    %     dataset. Remember to take care of corner cases.
    % distfunc: distance function to be used - l1, l2, linf.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    num_train = size(Xtrain, 1) ;
    num_test = size(Xtest, 1) ;
    labels = zeros(num_test, 1) ;  % empty vector for labels
    
    assert((K > 0) && (K < num_train + 1), "K must be between 1 and ...
    	number of training examples") ;
    
    for test = 1:num_test
        test_point = Xtest(test,:) ;
        
        if distfunc == "l1"
            test_dist = sum(abs(bsxfun(@minus, Xtrain, test_point)), 2) ;
        elseif distfunc == "l2"
            test_dist = sqrt(sum(bsxfun(@minus, Xtrain, test_point).^2, 2)) ;
        elseif distfunc == "linf"
            test_dist = max(abs(bsxfun(@minus, Xtrain, test_point)), [], 2) ;
        end
        
        % sort distances sin increasing order and get index of original spot in Xtrain
        [dist_sort, index_sort] = sort(test_dist) ; 
        
        k_nn_ind = index_sort(1 : K) ;  % get indices of k nearest neighbors
        k_nn = Ytrain(k_nn_ind) ;  % get y values for k NN
        
        if mean(k_nn) >= 0.5  % if tied, predict 1
            labels(test) = 1 ;
        else 
            labels(test) = 0 ;
    end
end

        
        
        
        
        
        