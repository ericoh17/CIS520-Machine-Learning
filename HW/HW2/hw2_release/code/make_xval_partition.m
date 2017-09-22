function [part] = make_xval_partition_jerry(n, n_folds)
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
%set seed
%rng(1111);

part = zeros(1,n);

if mod(n,n_folds)==0
    population = 1:n;
    for j=1:n_folds
        pick = randsample(population,n/n_folds);
        part(pick)=j;
        population = setdiff(population,pick);
    end
else
    population =1:n;
    size_1 = (n - mod(n,n_folds))/n_folds;
    rem_1 = mod(n,n_folds);
    onelarge = randsample(n_folds,rem_1);
    for j=1:n_folds
        if ismember(j,onelarge)==1
            pick = randsample(population,size_1+1);
        else
            pick = randsample(population,size_1);
        end
        part(pick)=j;
        population = setdiff(population,pick);
    end
end
    
