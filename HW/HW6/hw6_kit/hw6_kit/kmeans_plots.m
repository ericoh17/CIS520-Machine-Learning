
X = [2 1; 2 2; 3 1; 3 2;
    8 6; 7 7; 7 8; 8 9;
    12 6; 13 7; 13 8; 12 9;
    17 1; 17 2; 17 3];

%x1 = min(X(:,1)):0.01:max(X(:,1));
%x2 = min(X(:,2)):0.01:max(X(:,2));
%[x1G,x2G] = meshgrid(x1,x2);
%XGrid = [x1G(:),x2G(:)]; % Defines a fine grid on the plot

rng(23) ; 

%c_start = [7 7; 8 9; 12 9] ;
c_start = [12 6; 8 9; 12 9] ;
[idx1,C1] = kmeans(X, 3, 'MaxIter', 1, 'Start', c_start) ;

figure;
plot(X(idx1==1,1),X(idx1==1,2),'b.','MarkerSize',12);
hold on;
plot(X(idx1==2,1),X(idx1==2,2),'r.','MarkerSize',12);
plot(X(idx1==3,1),X(idx1==3,2),'g.','MarkerSize',12);
plot(c_start(:,1),c_start(:,2),'kx','MarkerSize',15,'LineWidth',3);
text(c_start(1,1)+0.25,c_start(1,2)-0.25,['(' num2str(c_start(1,1)) ',' num2str(c_start(1,2)) ')']) ;
text(c_start(2,1)+0.25,c_start(2,2)-0.25,['(' num2str(c_start(2,1)) ',' num2str(c_start(2,2)) ')']) ;
text(c_start(3,1)+0.25,c_start(3,2)-0.25,['(' num2str(c_start(3,1)) ',' num2str(c_start(3,2)) ')']) ;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
'Location','NorthEast');
hold off;

[idx2,C2] = kmeans(X, 3, 'MaxIter', 1, 'Start', C1) ;

figure;
plot(X(idx2==1,1),X(idx2==1,2),'b.','MarkerSize',12);
hold on;
plot(X(idx2==2,1),X(idx2==2,2),'r.','MarkerSize',12);
plot(X(idx2==3,1),X(idx2==3,2),'g.','MarkerSize',12);
plot(C1(:,1),C1(:,2),'kx','MarkerSize',15,'LineWidth',3);
text(C1(1,1)+0.25,C1(1,2)-0.25,['(' num2str(C1(1,1)) ',' num2str(C1(1,2)) ')']) ;
text(C1(2,1)+0.25,C1(2,2)-0.25,['(' num2str(C1(2,1)) ',' num2str(C1(2,2)) ')']) ;
text(C1(3,1)-1.3,C1(3,2)-0.25,['(' num2str(C1(3,1)) ',' num2str(C1(3,2)) ')']) ;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
'Location','NorthEast');
hold off;

[idx3,C3] = kmeans(X, 3, 'MaxIter', 1, 'Start', C2) ;

figure;
plot(X(idx3==1,1),X(idx3==1,2),'b.','MarkerSize',12);
hold on;
plot(X(idx3==2,1),X(idx3==2,2),'r.','MarkerSize',12);
plot(X(idx3==3,1),X(idx3==3,2),'g.','MarkerSize',12);
plot(C2(:,1),C2(:,2),'kx','MarkerSize',15,'LineWidth',3);
text(C2(1,1)-1,C2(1,2)-0.25,['(' num2str(C2(1,1)) ',' num2str(C2(1,2)) ')']) ;
text(C2(2,1)+0.25,C2(2,2)-0.25,['(' num2str(C2(2,1)) ',' num2str(C2(2,2)) ')']) ;
text(C2(3,1)-1.5,C2(3,2)-0.25,['(' num2str(C2(3,1)) ',' num2str(C2(3,2)) ')']) ;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
'Location','NorthEast');
hold off;

[idx4,C4] = kmeans(X, 3, 'MaxIter', 1, 'Start', C3) ;

figure;
plot(X(idx4==1,1),X(idx4==1,2),'b.','MarkerSize',12);
hold on;
plot(X(idx4==2,1),X(idx4==2,2),'r.','MarkerSize',12);
plot(X(idx4==3,1),X(idx4==3,2),'g.','MarkerSize',12);
plot(C3(:,1),C3(:,2),'kx','MarkerSize',15,'LineWidth',3);
text(C3(1,1)+0.25,C3(1,2)-0.25,['(' num2str(C3(1,1)) ',' num2str(C3(1,2)) ')']) ;
text(C3(2,1)+0.25,C3(2,2)-0.25,['(' num2str(C3(2,1)) ',' num2str(C3(2,2)) ')']) ;
text(C3(3,1)-1,C3(3,2)-0.25,['(' num2str(C3(3,1)) ',' num2str(C3(3,2)) ')']) ;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
'Location','NorthEast');
hold off;

[idx5,C5] = kmeans(X, 3, 'MaxIter', 1, 'Start', C4) ;

figure;
plot(X(idx5==1,1),X(idx5==1,2),'b.','MarkerSize',12);
hold on;
plot(X(idx5==2,1),X(idx5==2,2),'r.','MarkerSize',12);
plot(X(idx5==3,1),X(idx5==3,2),'g.','MarkerSize',12);
plot(C4(:,1),C4(:,2),'kx','MarkerSize',15,'LineWidth',3);
text(C4(1,1)+0.25,C4(1,2)-0.25,['(' num2str(C4(1,1)) ',' num2str(C4(1,2)) ')']) ;
text(C4(2,1)+0.25,C4(2,2)-0.25,['(' num2str(C4(2,1)) ',' num2str(C4(2,2)) ')']) ;
text(C4(3,1)-1,C4(3,2)-0.25,['(' num2str(C4(3,1)) ',' num2str(C4(3,2)) ')']) ;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
'Location','NorthEast');
hold off;


[idx6,C6] = kmeans(X, 3, 'MaxIter', 1, 'Start', C5) ;

figure;
plot(X(idx6==1,1),X(idx6==1,2),'b.','MarkerSize',12);
hold on;
plot(X(idx6==2,1),X(idx6==2,2),'r.','MarkerSize',12);
plot(X(idx6==3,1),X(idx6==3,2),'g.','MarkerSize',12);
plot(C5(:,1),C5(:,2),'kx','MarkerSize',15,'LineWidth',3);
text(C5(1,1)+0.25,C5(1,2)-0.25,['(' num2str(C5(1,1)) ',' num2str(C5(1,2)) ')']) ;
text(C5(2,1)+0.25,C5(2,2)-0.25,['(' num2str(C5(2,1)) ',' num2str(C5(2,2)) ')']) ;
text(C5(3,1)-1,C5(3,2)-0.25,['(' num2str(C5(3,1)) ',' num2str(C5(3,2)) ')']) ;
legend('Cluster 1','Cluster 2','Cluster 3','Centroids',...
'Location','NorthEast');
hold off;





