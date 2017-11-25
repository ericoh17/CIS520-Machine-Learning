
% script to clean data and run models for CIS520 Final Project

% load data
load('train.mat') ;
load('validation.mat') ;
%fileID = fopen('vocabulary.txt') ;
load Cell.mat ;

vocabulary = C;

% transform sparse matrix to full matrix
X_train_full = full(X_train_bag) ;
X_validation_full = full(X_validation_bag) ;

% remove feature 29 from data because it is empty
X_train_full(:, 29) = [];
X_validation_full(:, 29) = [];
vocabulary(29) = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% REMOVE MISSING FEATURES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check training and validation sets for missing features (ie. column of 0)
% find number of non-zero tweets for each feature (if 0, then maybe remove)
train_numCol_zero = sum(X_train_full~=0, 1) ; 
%valid_numCol_zero = sum(X_validation_full~=0, 1) ;

train_zeroFeat = sum(train_numCol_zero == 0) ;  % 440
%valid_zeroFeat = sum(valid_numCol_zero == 0) ;  % 1894

%%% currently wrong, need to select features missing in both training and testing sets %%%
% delete missing features (not sure if we want to do this)
X_train_full(:, ~any(train_numCol_zero, 1)) = [];
%X_validation_full(:, ~any(valid_numCol_zero, 1)) = [];
vocabulary(:, ~any(train_numCol_zero, 1)) = [] ;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% MANUAL TRUNCATION %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_vocab = length(vocabulary) ; 

for j=1:num_vocab
  vocabulary(j) = strtrim(vocabulary(j)) ;  
end

% remove feature 29 from data because it is empty
X_train_full(:, 29) = [];
X_validation_full(:, 29) = [];
vocabulary(29) = [];

num_vocab = length(vocabulary);

feature_remove = [] ;

for l=1:num_vocab
    
    % remove all double numbers
    if isa(vocabulary{l}, 'double') == 1
      feature_remove = [feature_remove, l] ;
    end
    
    % remove all string numbers
    if isstrprop(vocabulary{l}, 'digit') == 1
      feature_remove = [feature_remove, l] ;
    end
    
    % remove all punctuation
    if isstrprop(vocabulary{l}, 'punct') == 1
       feature_remove = [feature_remove, l] ;
    % remove all single letters
    elseif isa(vocabulary{l}, 'char') == 1 && length(vocabulary{l}) == 1
      feature_remove = [feature_remove, l] ;
    end
    
end

% remove features from training, validation data
X_train_full(:, feature_remove) = [];
%X_validation_full(:, feature_remove) = [];
vocabulary(feature_remove) = [];


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

% remove any bigrams with punctuation in them
num_bigram = length(all_bigram) ;
bigram_remove = [];

for f=1:num_bigram
  if any(isstrprop(all_bigram{f},'punct')) == 1
     bigram_remove = [bigram_remove, f] ; 
  end
  
  if any(isstrprop(all_bigram{f}, 'digit')) == 1
    bigram_remove = [bigram_remove, f] ;  
  end
  
  if isempty(strfind(all_bigram{f}, '<user>')) == 0
      bigram_remove = [bigram_remove, f] ; 
  end
  
end

all_bigram(bigram_remove) = [];

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
common_bigram = words(bin_idx > 5) ; 


% create bigram feature mat
num_tweet = size(X_train_full, 1) ; 
num_bigram_feat = length(common_bigram) ; 
bigram_feat = zeros(num_tweet, num_bigram_feat) ; 

for t=1:num_tweet
    for b=1:num_bigram_feat
        bigram_in_tweet = strfind(train_raw{t}, common_bigram{b}) ; 
        if isempty(bigram_in_tweet) == 1
            bigram_feat(t, b) = 0 ;
        else
          bigram_feat(t,b) = length(bigram_in_tweet) ;  
        end
    end
end

X_train_full = [X_train_full, bigram_feat] ; 

cov_train = cov(X_train_full);
[coeff_train, latent] = pcacov(cov_train);
score_train = X_train_full * coeff_train;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% Testing %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% multiclass logistic regression
addpath('./liblinear');

% run PCA
%[score_train, score_test, numpc] = pca_getpc(X_train_full, X_validation_full);

cov_train = cov(X_train_full);
[coeff_train, latent] = pcacov(cov_train);
score_train = X_train_full * coeff_train;
score_test = X_validation_full * coeff_train;
numpc = find(cumsum(latent)/sum(latent)>=0.9, 1) ;

% only keep PCs that give 90 percent reconstruction
score_train = score_train(:,1:numpc) ;
score_test = score_test(:,1:numpc) ;

logit_model = train(Y_train, sparse(score_train), ['-s 0', 'col']);

% save trained model
save('trained_logit_model','logit_model','coeff_train','numpc') ;


%%%%% CALCULATE CV ERROR FOR LOGISTIC REGRESSION %%%%%

% add column of ones to account for bias term
X_train_full = [ones(size(X_train_full, 1), 1), X_train_full] ; 

% generate folds
n = size(score_train, 1) ;
folds = make_partition(n,5);

N = max(folds) ;
error_mat = zeros(N, 1) ;

for i=1:N
  test_ind = find(folds == i);
  train_ind = setdiff(1:n, test_ind);

  train_feature = score_train(train_ind,:);
  train_label = Y_train(train_ind);
  test_feature = score_train(test_ind,:);
  test_label = Y_train(test_ind);

  % run sparse logistic regression
  model = train(train_label, sparse(train_feature), ['-s 0', 'col']);
  [predicted_label] = predict(test_label, sparse(test_feature), model, ['-q', 'col']);

  % calculate precision using given loss matrix
  error_mat(i) = performance_measure(predicted_label,test_label);

end

error = mean(error_mat) ; % 1.0044


load Cell.mat ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Naive-Bayes %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_train_full(:, 69) = [];

n = size(X_train_full, 1) ;
folds = make_partition(n,5);

test_ind = find(folds == 2);
train_ind = setdiff(1:n, test_ind);

train_feature = X_train_full(train_ind,:);
train_label = Y_train(train_ind);
test_feature = X_train_full(test_ind,:);
test_label = Y_train(test_ind);

% remove feature 69?
nb_mod = fitcnb(train_feature, train_label) ;


[n_t,~]=size(X_train_full);
[n_test,~]=size(X_validation_full);

wordlist =  strings(1,length(vocabulary));
for i=1:length(vocabulary)
    wordlist(i) = stem(char(vocabulary(i)));
end

[wordlist_u,ia,ic] = unique(wordlist);
X_test_new = zeros(n_test,length(wordlist_u));
X_t_new = zeros(n_t,length(wordlist_u));
for i=1:length(wordlist_u)
    idx = find(ic==i);
    X_test_new(:,i) = sum(X_test(:,idx),2);
    X_t_new(:,i) = sum(X_t(:,idx),2);
end



