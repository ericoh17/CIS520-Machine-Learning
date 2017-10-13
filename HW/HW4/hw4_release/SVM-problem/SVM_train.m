function [pred_labels] = SVM_train(test_data, kerneltype)
    % INPUT : 
    % test_data   - m X n matrix, where m is the number of test points and n is number of features
    % kerneltype  - one of strings 'poly', 'rbf'
    %               corresponding to polynomial, and RBF kernels
    %               respectively.
    
    % OUTPUT
    % returns a m X 1 vector predicted labels for each of the test points. The labels should be +1/-1 doubles

    
    % Default code below. Fill in your code on all the relevant positions

    m = size(test_data , 1);
    n = size(test_data, 2);

    
    %load cross-validation data
    
    C=[1,10,10^2,10^3,10^4,10^5];
    Q=[1,2,3,4,5];
    Sigma=[0.01,1,10,10^2,10^3];
    
    
    if strcmpi(kerneltype, 'poly')
        train_error = zeros(length(C),length(Q));
    elseif strcmpi(kerneltype, 'rbf')
        train_error = zeros(length(C),length(Sigma));
    else
        error('wrong kernel type');
    end
    
    
        for i=1:size(train_error,1)
            for j=1:size(train_error,2)
                errors = zeros(1,5);
                for fold=1:5
                    datadir=strcat('Breast-Cancer/CrossValidation/Fold',num2str(fold),'/cv-train.mat');
                    load(datadir);
                    train_data = cv_train;
                    train_feature = train_data(:,1:(end-1));
                    train_label = train_data(:,end);
                    
                    datadir=strcat('Breast-Cancer/CrossValidation/Fold',num2str(fold),'/cv-test.mat');
                    load(datadir);
                    test =cv_test;
                    test_feature = test(:,1:(end-1));
                    test_label = test(:,end);
                    if strcmpi(kerneltype, 'poly')
                        arg = horzcat('-t 1 -g 1 -r 1 -d ',num2str(Q(j)),' -c ',num2str(C(i)),' -q');
                    else
                        arg = horzcat('-t 2 -g ',num2str(Sigma(j)),' -c ',num2str(C(i)),' -q');
                    end
                    model = svmtrain(train_label, train_feature, arg);
                    [~, accuracy, ~] = svmpredict(test_label, test_feature, model);
                    errors(fold) = 1-accuracy(1)/100;
                end
                train_error(i,j) = mean(errors);
            end
        end
        
    A=min(train_error);
    min_error = min(A);
    [row,column]=find(train_error==min_error);
    
    datadir=strcat('Breast-Cancer/train.mat');
    load(datadir);
    train_data=train;
    train_feature = train_data(:,1:(end-1));
    train_label = train_data(:,end);
    
    C_opt=C(row);
    if strcmpi(kerneltype, 'poly')
        Q_opt = Q(column);
        arg = horzcat('-t 1 -g 1 -r 1 -d ',num2str(Q_opt),' -c ',num2str(C_opt),' -q');
    else
        Sigma_opt = Sigma(column);
        arg = horzcat('-t 2 -g ',num2str(Sigma_opt),' -c ',num2str(C_opt),' -q');
    end
    
    model = svmtrain(train_label, train_feature, arg);
    test_feature = test_data(:,1:(end-1));
    test_label = test_data(:,end);
    
    [pred_labels, ~, ~] = svmpredict(test_label, test_feature, model);



    % Do cross-validation
    % For all c
    % For all kernel parameters
    % Calculate the average cross-validation error for the 5-folds

    %your code


    %Train SVM on training data for the best parameters

    %your code


    % Do prediction on the test data
    % pred_labels = your prediction on the test data
    % your code














end
