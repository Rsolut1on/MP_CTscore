%% compute rate of volume change 
for i=1:186

    id_cur = V_les{i,1}(1:end-9);
    t_cur = V_les{i, 5};
    v_cur = V_les{i, 2} + V_les{i, 3} + V_les{i, 4};
    if isnan(t_cur)
        id_cur
        continue
    end
    if i == 1 || ~strcmp(id_pre,id_cur)
       t_pre = 0;
       v_pre = 0;
    end
    V_les{i, 6} = (v_cur - v_pre)/ (t_cur - t_pre);
    id_pre = id_cur;
    t_pre = t_cur;
    v_pre = v_cur;
end
%% compute ratio of sever add
data(data(:,2)<0,:)=[];
single2add = [];
exist_list=data(:,end);
for i=1:186
   cur_id = features(i,end);
   index = find(cur_id == exist_list);
   if ~length(index)
       single2add = [single2add;features(i,:)];
       exist_list = [exist_list; cur_id];
   end
    
end
data(:,2)=data(:,3)+data(:,7)+data(:,15);

cur_id = data(1,end);
pre_id = -1;
pre_temp = data(1,1:end-1);
% ratio_vs = zeros(186,1);
f_select = 7;
for i = 1:3
    temp = data(i,1:end-1);
    cur_id = data(i, end);
    if cur_id ~= pre_id
        pre_temp = temp;
        pre_id = cur_id;
        ratio_v = temp(f_select)/temp(1);
    else
        ratio_v = (temp(f_select)-pre_temp(f_select))/(temp(1)-pre_temp(1));
        pre_temp = temp;
    end
    
    ratio_vs(i,3) = ratio_v; 
end
