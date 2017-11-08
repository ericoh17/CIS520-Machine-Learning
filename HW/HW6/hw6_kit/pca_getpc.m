function [score_train, score_test, numpc] = pca_getpc(X_train, X_test)

% input: original X for training and testing
% output: PCAed X for training and testing, number of PCs that you selected

cov_train = cov(X_train);
[coeff_train, latent] = pcacov(cov_train);
score_train = X_train * coeff_train;
score_test = X_test * coeff_train;

figure, plot(cumsum(latent)/sum(latent));

% set you numpc here, you should acheive 90% reconstruction accuracy
% numpc = ?;

end