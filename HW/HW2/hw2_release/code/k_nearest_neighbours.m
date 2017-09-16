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
    
    % YOUR CODE GOES HERE.
    
    num_train = size(Xtrain, 1) ;
    num_test = size(Xtest, 1) ;
    
    labels = zeros(num_test, 1) ;
    
    for test = 1:num_test
        test_point = Xtest(test,:) ;
        test_dists = zeros(num_train, 1) ; 
        
        if distfunc == "l1"
            test_dist(test) = sum(abs(bsxfun(@minus, Xtrain, test_point)), 2) ;
        elseif distfunc == "l2"
            test_dist(test) = sqrt(sum(bsxfun(@minus, Xtrain, test_point).^2, 2)) ;
        elseif distfunc == "linf"
            test_dist(test) = max(abs(bsxfun(@minus, Xtrain, test_point)), [], 2) ;
        end
        
        dist_sort = sort(test_dist) ; 
        k_nn = dist_sort(1 : K) ; 
    end
end

        
        
        
        
        
        