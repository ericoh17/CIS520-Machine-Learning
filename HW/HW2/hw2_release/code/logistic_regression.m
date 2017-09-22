function labels = logistic_regression(Xtrain,Ytrain,Xtest,stepsize,iterations)

    % Function to perform logistic regression on the given data (binary classification)
    % Usage: labels = logistic_regression(Xtrain,Ytrain,Xtest)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (0/1)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % labels : return an M x 1 vector of predicted labels for testing data.
    
    % You may use gradient descent as a subroutine within this function to
    % make things simpler. Use the version of gradient descent
    % (constant step size, variable step size) and the step size which
    % you found to work best empirically.
    
    % Remember: Any modification you might wish to make on the
    % training & testing data must be done here (e.g. adding a new feature).
    % Remember: logistic regression will return probability values, and not
    % the actual labels themselves. This function has to return binary
    % labels, so you will have to perform some thresholding on the computed
    % probability values.
    
  
    
    %add on covarite
    Xtrain_ones = [ones(size(Xtrain,1),1) Xtrain];
    Xtest_ones = [ones(size(Xtest,1),1) Xtest];
    
    % FILL IN THE REST OF YOUR CODE.
    weights = ones(size(Xtrain_ones,2),1); % P x 1 vector of initial weights
    step_size = stepsize;
    n = size(Xtrain,1);
    
    for iter = [1:iterations]
        %step_size = step_size*1/sqrt(2); %step size changing
        
        p = exp(Xtrain_ones * weights)./(1+exp(Xtrain_ones*weights)); %probability
        gradiant = 1/n*Xtrain_ones'* (Ytrain - p);
        weights = weights + step_size * gradiant;
    end
    
    labels = exp(Xtest_ones * weights)./(1+exp(Xtest_ones * weights)) >=0.5;
    

end