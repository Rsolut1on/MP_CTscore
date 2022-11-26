%% get CT score
load('CTscore_12f_IDlist.mat')

data_norml(:,:) = data(:,:);
f_num = 24; 
gap_num = 10; 
gaps=[];
for i=1:f_num % hist for gap
    temp = data(:,i);
    temp_or = temp;
    [~,gap] = hist(temp,gap_num-1);
    temp(temp_or<=gap(1)) = 1;
    for g = 1:gap_num-2
    temp(gap(g)<temp_or&temp_or<=gap(g+1))=g+1;
    end
    temp(gap(end)<=temp_or)=gap_num;
    
    data_norml(:,i) = temp;
end
%% trian set/ test set
rng(1024,'twister');

rand_order=randperm(169);
data_tr = data(rand_order(1:118),:);
y_tr = y(rand_order(1:118),:);
data_te = data(rand_order(119:end),:);
y_te = y(rand_order(119:end),:);
%% staging RF num of features
num =20;
imps=zeros(num,f_num);
score_s=zeros(num,length(y_te),2);
accs = zeros(num,1);
for n_f = 1:num
    Model = TreeBagger(50,data_tr,y_tr,'Method','classification', 'OOBPredictorImportance', 'on');
    imp = Model.OOBPermutedPredictorDeltaError;
    imps(n_f,:) = imp;
    [predict_label,scores] = predict(Model, data_te);
    score_s(n_f,:,:) = scores;
    k=0;
    for i=1:length(predict_label)
        label = str2num(predict_label{i,1});
        if label==y_te(i)
            k = k+1;
        end

    end
    accs(n_f) = k/length(predict_label);
end
imps_mean = mean(imps, 1);
mean(accs)
%% top12
top_n =12;
[~,idx] = sort(imps_mean,'descend');
ids = idx(1:top_n);
we = imps_mean(ids);
we=we./sum(we);
CT_score = 0;
for i =1:length(we)
CT_score = CT_score + data_norml(:,i)*we(i);
end
CT_score= CT_score./(10*length(we));
%% imporved RF
select_id = [2,3,7,8,9,14,18,19,21,22,23,24];%picked features
ids = select_id-1;
imps=zeros(20,top_n);
score_s=zeros(num,length(y_te),2);
accs = zeros(20,1);
for i=1:20
Model = TreeBagger(50,data_tr(:,ids),y_tr,'Method','classification', 'OOBPredictorImportance', 'on');
imp = Model.OOBPermutedPredictorDeltaError;
imps(i,:) = imp;

[predict_label,scores] = predict(Model, data_te(:,ids));
score_s(n_f,:,:) = scores;
k=0;
for j=1:length(predict_label)
    label = str2num(predict_label{j,1});
    if label==y_te(j)
        k = k+1;
    end
end
acc = k/length(predict_label);
accs(i,1)=acc;
end
imps_mean_mini = mean(imps, 1);
mean(accs)
