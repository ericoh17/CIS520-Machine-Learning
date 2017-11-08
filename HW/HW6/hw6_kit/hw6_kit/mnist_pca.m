
% script to run PCA on part of MNIST dataset and cluster them

load('data/MNIST_train.mat') ;
load('data/MNIST_test.mat') ;

rng(147) ;

[PCAloadings, PCAscores, PCAvar, tsquared, explained] = pca(X_train) ;

%projection = X_train * PCAloadings ;
%projection0 = projection(Y_train == 1, :) ;
%projection1 = projection(Y_train == 2, :) ;

proj1 = PCAscores(Y_train==1,:);
proj2 = PCAscores(Y_train==2,:);

figure;
plot(proj1(:,1),proj2(:,1),'ob','MarkerSize',6);
hold on;
plot(proj1(:,2),proj2(:,2),'+m','MarkerSize',6);
xlabel('PC1') ;
ylabel('PC2') ;
title('Test digits for the first 2 PCA dimensions') ;
legend('PCA 1','PCA 2','Location','NorthEast');
hold off;

figure;
plot(projection0(:,1),projection1(:,1),'ob','MarkerSize',6);
hold on;
plot(projection0(:,2),projection1(:,2),'+m','MarkerSize',6);
xlabel('PC1') ;
ylabel('PC2') ;
title('Test digits for the first 2 PCA dimensions') ;
legend('PCA 1','PCA 2','Location','NorthEast');
hold off;

mu = mean(X_train) ;
nPC = size(PCAloadings, 2);

err_mat = zeros(nPC, 2);
err_mat(:,1) = 1:nPC;

for pcnum = 1:nPC 
  xhat = PCAscores(:,1:pcnum) * PCAloadings(:,1:pcnum)' ;
  xhat = bsxfun(@plus, xhat, mu) ;

  reconstruct_err = sqrt(sum(bsxfun(@minus, X_train, xhat).^2, 2)) ;
  
  err_mat(pcnum, 2) = mean(reconstruct_err) ;

end

figure;
plot(1:nPC, err_mat(:,2));
xlabel('Principal Components included') ;
ylabel('Average reconstruction error') ;
title({'Average reconstruction error as a function','of principal components included'}) ;


% find number of PCs needed to explain 85% of the variation
PCvariation = cumsum(explained) ;
minPC = find(PCvariation >= 85, 1) ;


% cluster PCs into 10 clusters with 100 PCs %
numdim = 100 ;
[idx, C] = kmeans(PCAscores(:,1:numdim), 10) ;

label = zeros(10, 2) ;
label(:,1) = 1:10 ;

for index = 1:10 
  label(index, 2) = mode(Y_train(idx == index));
end

test_center = bsxfun(@minus, X_test, mean(X_test)) ; 
project_test = test_center * PCAloadings(:,1:numdim) ;
err = k_means(PCAscores(:,1:numdim), Y_train, project_test, Y_test, 10);


% cluster PCs into 10 clusters  with 150 PCs %
numdim = 150 ;
[idx, C] = kmeans(PCAscores(:,1:numdim), 10) ;

label = zeros(10, 2) ;
label(:,1) = 1:10 ;

for index = 1:10 
  label(index, 2) = mode(Y_train(idx == index));
end

test_center = bsxfun(@minus, X_test, mean(X_test)) ; 
project_test = test_center * PCAloadings(:,1:numdim) ;
err = k_means(PCAscores(:,1:numdim), Y_train, project_test, Y_test, 10);


% cluster PCs into 10 clusters  with 200 PCs %
numdim = 200 ;
[idx, C] = kmeans(PCAscores(:,1:numdim), 10) ;

label = zeros(10, 2) ;
label(:,1) = 1:10 ;

for index = 1:10 
  label(index, 2) = mode(Y_train(idx == index));
end

test_center = bsxfun(@minus, X_test, mean(X_test)) ; 
project_test = test_center * PCAloadings(:,1:numdim) ;
err = k_means(PCAscores(:,1:numdim), Y_train, project_test, Y_test, 10);


% cluster PCs into 25 clusters with 100 PCs %
numdim = 100 ;
[idx, C] = kmeans(PCAscores(:,1:numdim), 25) ;

label = zeros(10, 2) ;
label(:,1) = 1:10 ;

for index = 1:10 
  label(index, 2) = mode(Y_train(idx == index));
end

test_center = bsxfun(@minus, X_test, mean(X_test)) ; 
project_test = test_center * PCAloadings(:,1:numdim) ;
err = k_means(PCAscores(:,1:numdim), Y_train, project_test, Y_test, 25);


% cluster PCs into 25 clusters with 150 PCs %
numdim = 150 ;
[idx, C] = kmeans(PCAscores(:,1:numdim), 25) ;

label = zeros(10, 2) ;
label(:,1) = 1:10 ;

for index = 1:10 
  label(index, 2) = mode(Y_train(idx == index));
end

test_center = bsxfun(@minus, X_test, mean(X_test)) ; 
project_test = test_center * PCAloadings(:,1:numdim) ;
err = k_means(PCAscores(:,1:numdim), Y_train, project_test, Y_test, 25);


% cluster PCs into 25 clusters with 200 PCs %
numdim = 200 ;
[idx, C] = kmeans(PCAscores(:,1:numdim), 25) ;

label = zeros(10, 2) ;
label(:,1) = 1:10 ;

for index = 1:10 
  label(index, 2) = mode(Y_train(idx == index));
end

test_center = bsxfun(@minus, X_test, mean(X_test)) ; 
project_test = test_center * PCAloadings(:,1:numdim) ;
err = k_means(PCAscores(:,1:numdim), Y_train, project_test, Y_test, 25);


