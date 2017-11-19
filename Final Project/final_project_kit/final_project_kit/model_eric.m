
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

%%% currently wrong, need to select features missing in both training and testing sets %%%
% delete missing features (not sure if we want to do this)
X_train_full(:, ~any(train_numCol_zero, 1)) = [];
X_validation_full(:, ~any(valid_numCol_zero, 1)) = [];


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

tbl = tabulate(all_bigram) ; 

freq = tbl(:,3) ; % get frequencies of each bigram
freq = cell2mat(freq) ; 

count = tbl(:,2) ; % get counts of each bigram
count = cell2mat(count) ; 

% find number of bigrams that occur more than once
num_singlebigram = find(count > 1, 1); 

% only extract the bigrams that occur more than once (at least twice)
[words,~,idx] = unique(all_bigram(:)) ;
bin_idx = hist(idx, unique(idx)) ;

% probably need to clean up features before this is meaningful
common_bigram = words(bin_idx > 1) ; 





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Testing %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% multiclass logistic regression
addpath('./liblinear');

% run PCA
[score_train, score_test, numpc] = pca_getpc(X_train_full, X_validation_full);

% only keep PCs that give 90 percent reconstruction
score_train = score_train(:,1:numpc) ;
score_test = score_test(:,1:numpc) ;

% run sparse logistic regression
model = train(Y_train, sparse(score_train), ['-s 0', 'col']);
[predicted_label] = predict(Y_test, sparse(X_test), model, ['-q', 'col']);





