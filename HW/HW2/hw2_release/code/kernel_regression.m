function labels = kernel_regression(Xtrain,Ytrain,Xtest,sigma)

    % Function that implements kernel regression on the given data (binary classification)
    % Usage: labels = kernel_regression(Xtrain,Ytrain,Xtest)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % sigma : width of the (gaussian) kernel.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    Ytrain(Ytrain == 0) = -1 ; % set 0 to -1 for convenience
    
    num_test = size(Xtest, 1) ;
    labels = zeros(num_test, 1) ;  % empty vector for labels
    
    scale = 1 / (2 * sigma^2) ; 
    
    for test = 1:num_test
        test_point = Xtest(test,:) ;
        
        kern = exp(-scale * (bsxfun(@minus, Xtrain, test_point).^2)) ;
        weight_kern = sum(kern .* y) ;
        
        if mode(sign(weight_kern)) == -1
            labels(test) = 0 ;  % set label to 0 to switch back from -1
        else 
            labels(test) = 1 ;
    end
end