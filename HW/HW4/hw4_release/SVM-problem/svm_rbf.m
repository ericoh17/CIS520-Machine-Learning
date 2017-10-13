addpath('C:\Users\Jerry Lu\Desktop\Upenn Study\Fall 2017\CIS 520\HW4\hw4_kit\hw4_release\SVM-problem\libsvm-3.22\libsvm-3.22\matlab')
%kernel svm using rbf kernel:
C=[1,10,10^2,10^3,10^4,10^5];
Sigma=[0.01,1,10,10^2,10^3];



C_opt = zeros(1,length(Sigma));
sigma_CV_err = zeros(length(Sigma), length(C)) ; 

for i=1:length(Sigma)
    test_error = zeros(5,length(C)); %for each fold and each C
 for fold=1:5
        datadir=strcat('Synthetic/CrossValidation/Fold',num2str(fold),'/cv-train.mat');
        train_data=load(datadir);
        train_data = train_data.cv_train;
        train_feature = train_data(:,1:(end-1));
        train_label = train_data(:,end);
        
        datadir=strcat('Synthetic/CrossValidation/Fold',num2str(fold),'/cv-test.mat');
        test_data=load(datadir);
        test_data = test_data.cv_test;
        test_feature = test_data(:,1:(end-1));
        test_label = test_data(:,end);
        
        
        for j=1:length(C)
            arg = horzcat('-t 2 -g ',num2str(Sigma(i)),' -c ',num2str(C(j)),' -q');
            model = svmtrain(train_label, train_feature, arg);
            [predict, accuracy, dec_values] = svmpredict(test_label, test_feature, model);
            test_error(fold,j) = 1-accuracy(1)/100;
        end
 end
    mean_test_error = mean(test_error);
    [~,index] = min(mean_test_error);
    C_opt(i) = C(index);
    
    sigma_CV_err(i,:) = mean_test_error ;
end

sigma_CV_err_opt = min(sigma_CV_err, [], 2);

train_error = zeros(1,length(Sigma));
test_error = zeros(1,length(Sigma));
 
for i=1:length(Sigma)
    datadir=strcat('Synthetic/train.mat');
    train_data=load(datadir);
    train_data = train_data.train;
    train_feature = train_data(:,1:(end-1));
    train_label = train_data(:,end);
    
    datadir=strcat('Synthetic/test.mat');
    test_data=load(datadir);
    test_data = test_data.test;
    test_feature = test_data(:,1:(end-1));
    test_label = test_data(:,end);
    
    
    arg = horzcat('-t 2 -g ',num2str(Sigma(i)),' -c ',num2str(C_opt(i)),' -q');
    model = svmtrain(train_label, train_feature, arg);
    [predict, accuracy, dec_values] = svmpredict(train_label, train_feature, model);
    train_error(i) = 1-accuracy(1)/100;
    
    [predict, accuracy, dec_values] = svmpredict(test_label, test_feature, model);
    test_error(i) = 1-accuracy(1)/100;
    
    %save the plot for decision boundary;
    %file = strcat('SVM_RBF_train_sigma_',num2str(Sigma(i)));
    %decision_boundary_SVM(train_feature, train_label, model, 200, file);
    %file = strcat('SVM_RBF_test_sigma_',num2str(Sigma(i)));
    %decision_boundary_SVM(test_feature, test_label, model, 200, file);
end
    
    % plot for 5.2
    figure;
    hold on
    plot(log10(Sigma),train_error,'b',log10(Sigma),test_error,'r');
    legend('training error', 'testing error');
    xlabel('log(sigma)');
    ylabel('CV error');
    hold off;

    filename = strcat('Plots/CV.png');
    title('CV Errors','FontSize', 16, 'FontWeight', 'bold');
    %print(h, '-dpdf', filename);
    saveas(gcf,filename,'png'); 
    
    % plot for 5.3
    figure;
    hold on
    plot(log10(Sigma),train_error,'b',log10(Sigma),test_error,'r', log10(Sigma), sigma_CV_err_opt, 'g' );
    legend('training error', 'testing error', 'CV error');
    xlabel('log(sigma)');
    ylabel('Classification error');
    hold off;

    filename = strcat('Plots/ClassificationErr_Sigma_CV','.pdf');
    title('Classification Error RBF Kernel','FontSize', 16, 'FontWeight', 'bold');
    %print(h, '-dpdf', filename);
    saveas(gcf,filename,'pdf');


[~,index] = min(test_error);
Sigma_opt = Sigma(index);
Sigma_opt
