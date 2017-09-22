function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE

size_fold = zeros(n_folds,1);  % empty vector of fold sizes
nDat = n;

% fill vector of fold sizes
for i =  1:n_folds 
    size_fold(i) = floor(nDat / (n_folds - i + 1)) ;  % divide number of vectors by remaining folds
    nDat = nDat - size_fold(i) ;  % updated number of vectors
end

part = zeros(1,n) ;
ind = 1 ;

ind_vec = 1:n_folds ; 

set_rng = randsample(100000, 1) ; 
rng(set_rng) ;

while length(ind_vec) > 0
     fold_ind = randsample(length(ind_vec),1) ;
     fold_num = ind_vec(fold_ind) ; 
     
     size_ind = size_fold(fold_num) ;
     part(ind : ind + size_ind - 1) = fold_num ;
     ind = ind + size_ind ;
     
     ind_vec(ind_vec == fold_num) = [];
     
end

part = part(randperm(length(part))) ;



