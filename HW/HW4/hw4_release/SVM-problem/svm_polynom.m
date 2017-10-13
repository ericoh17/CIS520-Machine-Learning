
% script to calculate CV error and test error for 
% differing values of C and q using a polynomial kernel

% set values of q and C to loop over
q_vec = [1, 2, 3, 4, 5] ;
C_vec = [1, 10, 10^2, 10^3, 10^4, 10^5];

C_opt = zeros(length(q_vec), 1) ; % C_opt = [100, 10, 1, 1, 1]
q_CV_err = zeros(length(q_vec), length(C_vec)) ; 

for k = 1:length(q_vec)
  
  degree = q_vec(k) ;
  fold_err = zeros(5, length(C_vec));
    
  for fold = 1:5
    
    load(strcat('Synthetic/CrossValidation/Fold',num2str(fold),'/cv-train.mat'));
    load(strcat('Synthetic/CrossValidation/Fold',num2str(fold),'/cv-test.mat'));
    
    for j = 1:length(C_vec)
      
      params = horzcat('-c ' ,num2str(C_vec(j)), ' -t 1 -g 1 -d ', num2str(degree)) ; 

      % train SVM on polynomial kernel
      poly_model = svmtrain(cv_train(:,end), cv_train(:,1:(end-1)), params) ;

      % test SVM on polynomial kernel
      [pred_label, accuracy, decision_val] = svmpredict(cv_test(:,end), cv_test(:,1:(end-1)), poly_model);

      % get classification error
      fold_err(fold, j) = classification_error(pred_label, cv_test(:,end)) ;  
      
    end
  end
  
  mean_cv_err = mean(fold_err) ;
  [~,ind] = min(mean_cv_err);
  C_opt(k) = C_vec(ind);
  
  q_CV_err(k,:) = mean_cv_err ;
  
end

q_CV_err_opt = min(q_CV_err, [], 2);

training_err = zeros(length(q_vec), 1) ; 
testing_err = zeros(length(q_vec), 1) ;

for l = 1:length(q_vec)
  
    load(strcat('Synthetic/train.mat'));
    load(strcat('Synthetic/test.mat'));
    
    degree = q_vec(l) ;
    
    % train SVM 
    params = horzcat('-c ' ,num2str(C_opt(l)), ' -t 1 -g 1 -d ', num2str(degree)) ; 
    poly_model = svmtrain(train(:,end), train(:,1:(end-1)), params) ;
    
    % test SVM on train data
    [pred_label, accuracy, decision_val] = svmpredict(train(:,end), train(:,1:(end-1)), poly_model);
    training_err(l) = classification_error(pred_label, train(:,end)) ;  
    
    % test SVM on test data
    [pred_label, accuracy, decision_val] = svmpredict(test(:,end), test(:,1:(end-1)), poly_model);
    testing_err(l) = classification_error(pred_label, test(:,end)) ;  
    
    %save the plot for decision boundary;
    file = strcat('SVM_polynomial_train_degree_',num2str(q_vec(l))) ;
    decision_boundary_SVM(train(:,1:(end-1)), train(:,end), poly_model, 200, file);
    
    file = strcat('SVM_polynomial_test_degree_',num2str(q_vec(l))) ;
    decision_boundary_SVM(test(:,1:(end-1)), test(:,end), poly_model, 200, file);
end

% plots for 5.2
figure;
hold on
plot(q_vec, training_err, 'b', q_vec, testing_err,'r');
legend('training error', 'testing error');
xlabel('degree(q)');
ylabel('Classification error');
hold off;

filename = strcat('Plots/ClassificationErr_degree','.pdf');
title('Classification Error Polynomial Kernel','FontSize', 16, 'FontWeight', 'bold');
%print(h, '-dpdf', filename);
saveas(gcf,filename,'pdf');



% plots for 5.3
figure;
hold on
plot(q_vec, training_err, 'b', q_vec, testing_err,'r', q_vec, q_CV_err_opt, 'g');
legend('training error', 'testing error', 'CV error');
xlabel('degree(q)');
ylabel('Classification error');
hold off;

filename = strcat('Plots/ClassificationErr_degree_CV','.pdf');
title('Classification Error Polynomial Kernel','FontSize', 16, 'FontWeight', 'bold');
%print(h, '-dpdf', filename);
saveas(gcf,filename,'pdf');

[~,ind] = min(testing_err);
q_opt = q_vec(ind); % 4
