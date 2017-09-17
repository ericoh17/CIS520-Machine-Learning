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

% set number of folds
N_folds = [2,4,8,16];
num_fold = length(N_folds) ;

% vectors to hold knn errors
errors_xval_knn = zeros(100, num_fold) ;
erros_xval_knn_noisy = zeros(100, num_fold) ;

% vectors to hold kern reg errors
errors_xval_kern = zeros(100, num_fold) ;
erros_xval_kern_noisy = zeros(100, num_fold) ;

errors_test = zeros(100, num_fold) ;

% set K=1 and sigma=1 for first set of simulations
K = 1;
sigma = 1 ;

distfunc = string('l2') ; 

for trial = 1:100
    for i = 1:num_fold
        fold = N_folds(i) ;
        part = make_xval_partition(N, fold) ;
        
        errors_xval_knn(i) = knn_xval_error(X, Y, K, part, distfunc) ;
        errors_xval_kern(i) = kernreg_xval_error(X, Y, sigma, part) ;
    end
end

% code to plot the error bars. change these values depending on what
% experiment you are running
y = mean(errors_xval_knn); e = std(errors_xval_knn); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_test); e = std(errors_test); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, N = [2,4,8,16]');
xlabel('N');
ylabel('Error');
legend('N-Fold Error','Test Error');
hold off;