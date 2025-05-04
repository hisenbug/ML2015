%% Classiftying Iris Data Using k-Nearest Neighbour Classifier

% Reading the Iris data
data1 = xlsread('D:\Workspace\Pattern_Recognition\Iris3\irisdata.xls');
class = ones(150,1);
class(51:100) = 2;
class(101:150) = 3;
data = [data1 class];
% Randomly selecting 15 Training data points from each of the 3 Classes
train = zeros(1,45);
for i=1:3
    k=1;
    while k <= 15
        r = (i-1)*50 + randi(50,1);
        f=0;
        for l=(i-1)*15+1:i*15
            if(train(l)==r)
                f=1;
                break;
            end
        end
        if f==0
            train((i-1)*15+k)=r;
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
    c1=0; c2=0; c3=0;
    dist = 100*ones(1,k);
    pos = ones(k);
    % Computing the k Nearest Neighbours
    for l=1:45
        d = sum((data(train(l),1:4)- data(test(i),1:4)).^2); %distance
        for m=1:k
            if d<dist(m)
                for n=k:-1:m+1
                    dist(n)=dist(n-1);
                    pos(n)=pos(n-1);
                end
                dist(m)=d;
                pos(m)=train(l);
                break;
            end
        end
    end
    % Counting Neighbours from each class
    for m=1:k
        if data(pos(m),5)==1
            c1 = c1+1;
        else if data(pos(m),5)==2
                c2 = c2+1;
            else
                c3 = c3+1;
            end
        end
    end
    % Classification
    if c1>c2 && c1>c3
        res(2,i)=1;
    else if c2>c1 && c2>c3
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