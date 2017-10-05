function model [pred_labels] = SVM_train(test_data, kerneltype)
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

    %load train_data

    datadir = 'Breast-Cancer/';

    load(strcat(datadir,'train.mat'));


    %load cross-validation data

    %your code




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
