% Submit your textual answers, and attach these plots in a latex file for
% this homework. 
% This script is merely for your convenience, to generate the plots for each
% experiment. Feel free to change it, as you do not need to submit this
% with your code.

% Loading the data: this loads X, and Ytrain.
load('../data/X.mat'); % change this to X_noisy if you want to run the code on the noisy data
load('../data/Y.mat');

N_folds = [2,4,8,16];
errors_xval = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the [N_folds(j)]-fold cross validation error in trial i 
errors_test = zeros(100,size(N_folds,2)); % errors_xval(i,j) records the true test error in trial i (the entire row will be identical)

for trial = 1:100
    
    % fill in the rest of your code here. 
    
end

% code to plot the error bars. change these values depending on what
% experiment you are running
y = mean(errors_xval); e = std(errors_xval); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
hold on;
y = mean(errors_test); e = std(errors_test); x = [2,4,8,16]; % <- computes mean across all trials
errorbar(x, y, e);
title('Original data, N = [2,4,8,16]');
xlabel('N');
ylabel('Error');
legend('N-Fold Error','Test Error');
hold off;