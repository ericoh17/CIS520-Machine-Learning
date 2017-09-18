% Submit your textual answers, and attach these plots in a latex file for
% this homework. 
% This script is merely for your convenience, to generate the plots for each
% experiment. Feel free to change it, as you do not need to submit this
% with your code.

% Loading the data: this loads X, and Ytrain.
load('../data/X.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('../data/Y.mat'); 

% split data into 450 training and 150 testing points
N = size(X,1) ;

tf = false(N, 1) ;
tf(1:450) = true ;
tf = tf(randperm(N)) ;

Xtrain = X(tf, :) ;
Xtest = X(~tf, :) ;
Ytrain = Y(tf) ;
Ytest = Y(~tf) ;

%%%% PLOT ERRORS FOR FIXED K AND SIGMA AND VARY FOLD NUM %%%

% set number of folds
num_fold = [2,4,8,16];
num_fold = length(num_fold) ;

% vectors to hold CV and test errors
errors_xval_knn = zeros(100, num_fold) ;
errors_xval_kern = zeros(100, num_fold) ;
errors_test_knn = zeros(100, 1) ;
errors_test_kern = zeros(100, 1) ;

% set K=1 and sigma=1 for first set of simulations
K = 1;
sigma = 1 ;
distfunc = '12' ; 

for trial = 1:100
    
    % obtain cross-validation errors
    for i = 1:num_fold
        fold = num_fold(i) ;
        part = make_xval_partition(N, fold) ;
        
        errors_xval_knn(trial, i) = knn_xval_error(Xtrain, Ytrain, K, part, distfunc) ;
        errors_xval_kern(trial, i) = kernreg_xval_error(Xrain, Ytrain, sigma, part) ;
    end
    
    % obtain true test errors
    knn_test_pred = k_nearest_neighbours(Xtrain, Ytrain, Xtest, K, distfunc) ;
    kern_test_pred = kernel_regression(Xtrain, Ytrain, Xtest, sigma) ;
    errors_test_knn(trial) = mean(knn_test_pred ~= Ytest) ;
    errors_test_kern(trial) = mean(kern_test_pred ~= Ytest) ;
    
end

% plot errors - change to noisy versions to plot errors from that dat
y = mean(errors_xval_knn); e = std(errors_xval_knn); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_test_knn); e = std(errors_test_knn); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, N = [2,4,8,16]');
% title('Noisy data, N = [2,4,8,16]');
xlabel('N');
ylabel('Error');
legend('N-Fold Error','Test Error');
hold off;


%%%% PLOT ERRORS FOR VARYING K and SIGMA AND FIXED 10 FOLDS %%%%

% set varying K and sigma
k_vec = [1 2 3 5 8 13 21 34] ;
num_k = length(k_vec) ;

sigma_vec = 1:12 ;
num_sig = length(sigma_vec) ;

num_fold = 10 ;

% vectors to hold CV and test errors
errors_xval_knn = zeros(100, num_k) ;
errors_xval_kern = zeros(100, num_sig) ;

errors_test_knn = zeros(100, num_k) ;
errors_test_kern = zeros(100, num_sig) ;

for trial = 1:100
    
    part = make_xval_partition(N, num_fold) ;
    
    % obtain cross-validation errors
    for j = 1:num_k
        K = k_vec(j) ;
        
        errors_xval_knn(trial, j) = knn_xval_error(Xtrain, Ytrain, K, part, distfunc) ;  % CV error
        knn_test_pred = k_nearest_neighbours(Xtrain, Ytrain, Xtest, K, distfunc) ;  
        errors_test_knn(trial, j) = mean(knn_test_pred ~= Ytest) ;  % test error
    end
    
    for l = 1:num_sig
       sigma = sig_vec(l) ;
       
       errors_xval_kern(trial, l) = kernreg_xval_error(Xrain, Ytrain, sigma, part) ;   % CV error
       kern_test_pred = kernel_regression(Xtrain, Ytrain, Xtest, sigma) ;
       errors_test_kern(trial, l) = mean(kern_test_pred ~= Ytest) ;   % test error
    end
    
end

% plot errors for KNN - change to noisy versions to plot errors from that dat
y = mean(errors_xval_knn); e = std(errors_xval_knn); x = [1 2 3 5 8 13 21 34]; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_test_knn); e = std(errors_test_knn); x = [1 2 3 5 8 13 21 34]; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, K = [1 2 3 5 8 13 21 34]');
%title('Noisy data, K = [1 2 3 5 8 13 21 34]');
xlabel('K');
ylabel('Error');
legend('10-Fold Error','Test Error');
hold off;


% plot errors for KernReg - change to noisy versions to plot errors from that dat
y = mean(errors_xval_kern); e = std(errors_xval_kern); x = 1:12 ; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_test_kern); e = std(errors_test_kern); x = 1:12 ; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, Sigma = 1-12');
%title('Noisy data, Sigma = 1-12');
xlabel('Sigma');
ylabel('Error');
legend('10-Fold Error','Test Error');
hold off;



