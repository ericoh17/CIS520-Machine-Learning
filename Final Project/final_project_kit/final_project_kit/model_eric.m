
% script to clean data and run models for CIS520 Final Project

% load data
load('train.mat') ;
load('validation.mat') ;
%fileID = fopen('vocabulary.txt') ;

% transform sparse matrix to full matrix
X_train_full = full(X_train_bag) ;
X_validation_full = full(X_validation_bag) ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% REMOVE MISSING FEATURES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check training and validation sets for missing features (ie. column of 0)
% find number of non-zero tweets for each feature (if 0, then maybe remove)
train_numCol_zero = sum(X_train_full~=0, 1) ; 
valid_numCol_zero = sum(X_validation_full~=0, 1) ;

train_zeroFeat = sum(train_numCol_zero == 0) ;  % 440
valid_zeroFeat = sum(valid_numCol_zero == 0) ;  % 1894

% delete missing features (not sure if we want to do this)
train_numCol_zero(:, ~any(train_numCol_zero, 1)) = [];
valid_numCol_zero(:, ~any(valid_numCol_zero, 1)) = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Autoencoder %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Does not work without 0/1 features ; need separate neural network toolbox

% train an autoencoder for potential dimension reduction
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');

% train autoencoder
dbn = rbm(X_train_full) ;

% learn new features
[new_feat_rbm, new_feat_test_rbm] = newFeature_rbm(dbn, X_train_full, X_validation_full) ; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% PCA %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pca on full training data
[coeff,score,latent, tsquared, explained] = pca(X_train_full);

% find number of PCs needed to explain 90% of the variation
PCvariation = cumsum(explained) ;
minPC = find(PCvariation >= 90, 1) ;

nPC = size(coeff, 2);
plot(1:nPC, PCvariation) ; 
grid on;
xlabel('Number of PCs') ;
ylabel('Explained variance') ; 


% pca on combined training and validation data for more data
all_X = vertcat(X_train_full, X_validation_full) ; 

[coeff,score,latent, tsquared, explained] = pca(all_X);

% find number of PCs needed to explain 90% of the variation
PCvariation = cumsum(explained) ;
minPC = find(PCvariation >= 90, 1) ;

nPC = size(coeff, 2);
plot(1:nPC, PCvariation) ; 
grid on;
xlabel('Number of PCs') ;
ylabel('Explained variance') ; 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% bi-gram for each tweet %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

all_bigram = [] ;
for i=1:size(train_raw, 1)
    bigram = ngrams(train_raw(i), 2) ;
    all_bigram = [all_bigram, bigram];  % very inefficient
end

[words,~,idx] = unique(all_bigram(:)) ;
bin_idx = hist(idx, unique(idx)) ;

% probably need to clean up features before this is meaningful
common_bigram = words(bin_idx > 1) ; 




