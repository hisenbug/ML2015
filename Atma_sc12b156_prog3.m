%% Classifying Iris Data Using k-Nearest Neighbour Classifier

% Reading the Iris data
data1 = xlsread('D:\Workspace\Pattern_Recognition\Iris3\irisdata.xls');
class = ones(150,1);
class(51:100) = 2;
class(101:150) = 3;
data = [data1 class];

% Applying PCA on the given data
md = data(:,1:4);
M = mean(md);
for i=1:150
    md(i,:) = md(i,:) - M;
end
[U,S,V] = svd(md);
% V is the matrix having eigenvectors as columns
n = input('Enter the Dimension of projection (1-4): ');
e = V(1:4,1:n);
a = md*e;
x = mean(1:n) + a*e(1:n,:);

% Randomly selecting 25 Training data points from each of the 3 Classes
train = zeros(1,75);
for i=1:3
    k=1;
    while k <= 25
        r = (i-1)*50 + randi(50,1);
        f=0;
        for l=(i-1)*25+1:i*25
            if(train(l)==r)
                f=1;
                break;
            end
        end
        if f==0
            train((i-1)*25+k)=r;
            k=k+1;
        end
    end
end
% Randomly selecting 15 Test samples from each of the 3 Classes
% (non-overlapping with the Training Set)
test = zeros(1,45);
for i=1:3
    k=1;
    while k <= 15
        r = (i-1)*50 + randi(50,1);
        f=0;
        for l=(i-1)*15+1:i*15
            if(train(l)==r || test(l)==r)
                f=1;
                break;
            end
        end
        for l=(i-1)*25+16:i*25
            if(train(l)==r)
                f=1;
                break;
            end
        end
        if f==0
            test((i-1)*15+k)=r;
            k=k+1;
        end
    end
end

k = input('Enter value of k for k-NN Classifier: ');
res = [zeros(1,45);zeros(1,45)];
for i=1:45
    res(1,i)=data(test(i),5);
    dist = 100*ones(1,3*k);
    pos = ones(3*k);
    % Computing the k Nearest Neighbours of each Class
    for z=1:3
        for l=(z-1)*25+1:z*25
            d = sum((x(train(l),1:n)- x(test(i),1:n)).^2); %distance
            for m=(z-1)*k+1:z*k
                if d<dist(m)
                    for q=z*k:-1:(z-1)*25+m+1
                        dist(q)=dist(q-1);
                        pos(q)=pos(q-1);
                    end
                    dist(m)=d;
                    pos(m)=train(l);
                    break;
                end
            end
        end
    end
    % Classification
    if dist(k)<dist(2*k) && dist(k)<dist(3*k)
        res(2,i)=1;
    else if dist(2*k)<dist(k) && dist(2*k)<dist(3*k)
            res(2,i)=2;
        else 
            res(2,i)=3;
        end
    end
end

% Making the Confusion Matrix
conf = zeros(3,3);
for i=1:45
    conf(res(1,i),res(2,i)) = conf(res(1,i),res(2,i))+1;
end
display('The Confusion Matrix is:');
display(conf);
acc = trace(conf)/45*100; %accuracy
display(sprintf('Classification Accuracy = %g', acc));

%% Support Vector Machine

% Reading the Iris data
data1 = xlsread('D:\Workspace\Pattern_Recognition\Iris3\irisdata.xls');
class = ones(150,1);
class(51:100) = 2;
class(101:150) = 3;
data = [data1 class];

train = zeros(1,75); test = zeros(1,45);
train1 = zeros(75,5); test1 = zeros(45,5);
% Randomly selecting 25 Training data points from each of the 3 Classes
train = zeros(1,75);
for i=1:3
    k=1;
    while k <= 25
        r = (i-1)*50 + randi(50,1);
        f=0;
        for l=(i-1)*25+1:i*25
            if(train(l)==r)
                f=1;
                break;
            end
        end
        if f==0
            train((i-1)*25+k)=r;
            train1((i-1)*25+k,:)=data(r,:);
            k=k+1;
        end
    end
end
% Randomly selecting 15 Test samples from each of the 3 Classes
% (non-overlapping with the Training Set)
for i=1:3
    k=1;
    while k <= 15
        r = (i-1)*50 + randi(50,1);
        f=0;
        for l=(i-1)*15+1:i*15
            if(train(l)==r || test(l)==r)
                f=1;
                break;
            end
        end
        for l=(i-1)*25+16:i*25
            if(train(l)==r)
                f=1;
                break;
            end
        end
        if f==0
            test((i-1)*15+k)=r;
            test1((i-1)*15+k,:)=data(r,:);
            k=k+1;
        end
    end
end

% Class 1 and Class 2
% Traing the SVM
svmS = svmtrain(train1(1:50,1:2), train1(1:50,5), 'showplot', true);
% Classifying data based on trained SVM
svmC = svmclassify(svmS, test1(1:30,1:2), 'showplot', true);
xlabel('sepal length in cm');
ylabel('sepal width in cm');
title('Classification b/w Class 1 and 2'); 
disp('Confusion Matrix:'); 
disp(confusionmat(test1(1:30,5),svmC));
acc = sum(test1(1:30,5)==svmC)/length(svmC); 
display(sprintf('Classification Accuracy = %g', 100*acc));

% Class 2 and Class 3
% Traing the SVM
figure,
svmS = svmtrain(train1(26:75,1:2), train1(26:75,5), 'showplot', true);
% Classifying data based on trained SVM
svmC = svmclassify(svmS, test1(16:45,1:2), 'showplot', true);
xlabel('sepal length in cm');
ylabel('sepal width in cm');
title('Classification b/w Class 2 and 3');
disp('Confusion Matrix:');
disp(confusionmat(test1(16:45,5),svmC));
acc = sum(test1(16:45,5)==svmC)/length(svmC); 
display(sprintf('Classification Accuracy = %g', 100*acc));
