function [weights,error_per_iter] = gradient_ascent_decay(Xtrain,Ytrain,initial_step_size,iterations)

    % Function to perform gradient descent with a decaying step size for
    % logistic regression.
    % Usage: [weights,error_per_iter] = gradient_descent(Xtrain,Ytrain,step_size,iterations)
    
    % The parameters to this function are exactly the same as the
    % parameters to gradient descent with fixed step size.
    
    % initial_step_size : This parameter refers to the initial value of the step
    % size. The actual step size to update the weights will be a value
    % that is (initial_step_size * some function that decays over time)
    % some good choices for this function might by 1/n or 1/sqrt(n).
    % Experiment with such functions, and initial step size until you get
    % good performance.
    
    %initial_step_size = 2e-3;
    %iterations=400;
    
    weights = ones(size(Xtrain,2),1); % P x 1 vector of initial weights
    error_per_iter = zeros(iterations,1); % error_per_iter(i) records training error in iteration i of GD.
    % dont forget to update these values within the loop!
    n = size(Xtrain,1);
    b =glmfit(Xtrain,Ytrain,'binomial','link','logit','constant','off');
    
    % FILL IN THE REST OF THE CODE %
    step_size = initial_step_size;
    for iter = [1:iterations]
        step_size = step_size*1/sqrt(1.1); %step size changing
        
        p = exp(Xtrain * weights)./(1+exp(Xtrain*weights)); %probability
        gradiant = 1/n*Xtrain'* (Ytrain - p);
        weights = weights + step_size * gradiant;
        y_predict = exp(Xtrain * weights)./(1+exp(Xtrain*weights)) >=0.5;
        error = y_predict ~= Ytrain;
        %error = norm(weights - b,2);
        error_per_iter(iter) = mean(error);
    end


end

