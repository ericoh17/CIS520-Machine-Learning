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

% set seed
rng(893) ;

size_fold = zeros(n_folds,1);  % empty vector of fold sizes
nDat = n;

% fill vector of fold sizes
for i =  1:n_folds 
    size = floor(nDat / (n_folds - i + 1)) ;  % divide number of vectors by remaining folds
    size_fold(i) = size ;
    nDat = nDat - size ;  % updated number of vectors
end

part = zeros(n,1) ;
ind = 1 ;

% get indices of datapoints
for j = 1:n_folds
    size_ind = size_fold(j) ;
    part(ind : ind + size_ind - 1) = j ;
    ind = ind + size_ind ;
end

part = part(randperm(length(part))) ;



