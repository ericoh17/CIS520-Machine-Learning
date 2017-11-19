cd 'C:\Users\Jerry Lu\Dropbox\CIS 520\Final Project\final_project_kit\final_project_kit'
addpath('C:\Users\Jerry Lu\Dropbox\CIS 520\Final Project\final_project_kit\final_project_kit\libsvm-3.22\libsvm-3.22\matlab')
%% load data
load('train.mat')
load('validation.mat')
load('vocabulary.mat')
%% transform sparse matrix to full matrix
X_t = full(X_train_bag);
X_v = full(X_validation_bag);

%% cleaning
% stemming
[n_t,~]=size(X_t);
wordlist =  strings(1,length(vocabulary));
for i=1:length(vocabulary)
    wordlist(i) = stem(char(vocabulary(i)));
end
[wordlist_u,ia,ic] = unique(wordlist);
X_t_new = zeros(n_t,length(wordlist_u));
for i=1:length(wordlist_u)
    idx = find(ic==i);
    X_t_new(:,i) = sum(X_t(:,idx),2);
end

% compute freq, pick one from 200 to 2000.
freq_t = sum(X_t_new);
feature_id = find(freq_t>=200 & freq_t <= 2000);

%creat feature
X_select = X_t_new(:,feature_id);

%% models: multistage
% generate folds
folds=make_partition(n_t,5);

%pca
[coeff,score,latent] = pca(X_select);
X_pca = score(:,1:100);
%test_x =  (X_test - ones(10000,1)*mean(X_test))*coeff(:,1:100);
%test_y = Y_test;

%first classify as positive and negative, use svm in the first stage
% cross validation to learn model
Y_train_strat = double((Y_train==1 | Y_train==3));
[q_opt,C_opt,test_error]=svm_poly(folds,X_pca,Y_train_strat);
Y_positive = Y_train(Y_train==1 | Y_train==3);
X_positive = X_pca(Y_train==1 | Y_train==3,:);
% SVM took to much time

%% k-means
[n,~] = size(X_select);
test_ind = find(folds==2);
train_ind = setdiff(1:n,test_ind);
        
train_feature = X_select(train_ind,:);
train_label = Y_train(train_ind);
test_feature = X_select(test_ind,:);
test_label = Y_train(test_ind);
[precision] = k_means(train_feature,train_label,test_feature,test_label , 5);
% CV-error = 1.3-1.4
%% boosting
addpath('C:\Users\Jerry Lu\Dropbox\CIS 520\Final Project\final_project_kit\final_project_kit\boosting')
nbIterations = 1000;
[n,~] = size(X_select);
test_ind = find(folds==2);
train_ind = setdiff(1:n,test_ind);
        
train_feature = X_select(train_ind,:);
train_label = Y_train_strat(train_ind);
train_label(train_label==0)=-1;
test_feature = X_select(test_ind,:);
test_label = Y_train_strat(test_ind);
test_label(test_label==0)=-1;

[classifiers, classifiersWeights] = adaBoostTrain(train_feature, train_label , nbIterations);
preds = adaBoostPredict(test_feature, classifiers, classifiersWeights);
error = computeError(preds, test_label);

addpath('C:\Users\Jerry Lu\Dropbox\CIS 520\Final Project\final_project_kit\final_project_kit\boosting\prtools4.2.5\prtools')
train_feature = X_select(train_ind,:);
train_label = Y_train(train_ind);
test_feature = X_select(test_ind,:);
test_label = Y_train(test_ind);
nbIterations = 10;
[classifiers, classifiersWeights] = adaBoostM1Train(train_feature, train_label , nbIterations);
y_pred = adaBoostM1Predict(test_feature, 5, classifiers, classifiersWeights);
err = performance_measure(y_pred, test_label);
%%


