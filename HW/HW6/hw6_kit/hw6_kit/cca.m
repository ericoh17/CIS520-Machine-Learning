
% script to read in breast cancer data and
% calculate largest singular vectors in CCA

load('data/breast_cancer.mat')
rng(957) ; 

X = X_train ;
Y = Y_train ;
Z = (X.' * X)^(-0.5) * (X.' * Y) * (double(Y.') * double(Y))^(-0.5) ;

[U, S, V] = svds(Z) ; 
corr((X * U), (Y * V)) ;

[PCAloadings, PCAscores, PCAvar] = pca(X) ;
betaPCR = regress(Y, PCAscores(:,1)) ;
ypred = PCAscores(:,1) * betaPCR ; 

corr(ypred, Y);





