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
    
    num_test = size(Xtest, 1) ;
    labels = zeros(num_test, 1) ;  % empty vector for labels 
    
    for test = 1:num_test
        test_point = Xtest(test,:) ;
        
        % calculate L2 distance
        dist = sqrt(sum(bsxfun(@minus, Xtrain, test_point).^2, 2)) ;
        
        kern = exp(-dist.^2/sigma.^2) ;
        weight_kern = sum(kern .* Ytrain)/sum(kern) ;
        
        if weight_kern >= 0.5 
            labels(test) = 1 ;
        else 
            labels(test) = 0 ;
        end
    end
end
