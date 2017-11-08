%% load data

load ./data/ocr_train.mat
load ./data/ocr_test.mat

%% pca

[score_train, score_test, numpc] = pca_getpc(X_train, X_test);

% your code to select new features using PCA-ed data



%% auto encoder

addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
 
% your code to train an Auto-encoder, then learn new features from the original data set
% use rbm.m and newFeature_rbm.m



%% logistic

addpath('./liblinear');
precision_ori_log = logistic(X_train, Y_train, X_test, Y_test);

% your code to train logistic on PCA-ed and Auto-encoder data



%% kmeans

K = [26, 50];
precision_ori_km = zeros(length(K), 1);
for i = 1:length(K)
    k = K(i);
    precision_ori_km(i) = k_means(X_train, Y_train, X_test, Y_test, k);
    
    % your code to train logistic on PCA-ed and Auto-encoder data
    
    
    
end
