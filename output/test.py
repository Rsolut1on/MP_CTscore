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

import os
import numpy as np
import pandas as pd
import csv
import datetime


root = 'LabCount'
LABEL_NAME = ['label_Exu.csv', 'label_Nod.csv', 'label_Hol.csv', 'lung_area.csv', 'cause_lung9.csv']

name_list, patients_data = [], []
for path, dirs, files in os.walk(root):
    if dirs and len(dirs[0]) == 8:
        dir_list = []

        for dir in dirs:
            dir_list.append(dir)
        name_list.append([path, dir_list])


for step, ele in enumerate(name_list):
    patient_data = []
    patient_id = ele[0][9:]
    admission_str = '0'

    for date in ele[1]:
        # lung
        lung_area = 0.0
        file_name = os.path.join(ele[0], date, LABEL_NAME[3])
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for e in reader:
                lung_area += float(e[2][2:-2])

        lung9_features = 0

    #exu
        exu_area2lung = 0.0
        file_name = os.path.join(ele[0], date, LABEL_NAME[0])
        datas = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for e in reader:
                e[0] = e[0][2:]
                e[-1] = e[-1][:-2]
                datas.append(e)
        if datas:
            datas = np.array(datas, dtype=np.float)
            exu_HUs1 = datas[:, 2]
            exu_HUs0 = datas[:, 3]
            exu_HU_max = np.max(exu_HUs1)
            exu_HU_min = np.min(exu_HUs0)
            exu_area2lung = np.sum(datas[:, 1]) / lung_area
            if exu_HU_max > 100:
                print()
        else:
            exu_area2lung = 0
            exu_size_max = 0
            exu_size_median = 0
            exu_size_min = 0
            exu_size_mean = 0
            exu_HU_max = None
            exu_HU_min = None

    #nodule
        nod_num = 0
        file_name = os.path.join(ele[0], date, LABEL_NAME[1])
        datas = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for e in reader:
                e[0] = e[0][2:]
                e[-1] = e[-1][:-2]
                datas.append(e)
        if datas:
            datas = np.array(datas, dtype=np.float)
            nod_areas = datas[:, 1]
            nod_areas_list = nod_areas.tolist()
            max_area_id = nod_areas_list.index(max(nod_areas))
            nod_HUs = datas[:, 2:]
            nod_size_max = max(nod_areas)
            nod_size_min = min(nod_areas)
            nod_size_mean = np.mean(nod_areas)
            nod_size_median = np.median(nod_areas)
            nod_HU_max = nod_HUs[max_area_id, 0]
            nod_HU_min = nod_HUs[max_area_id, 1]
            nod_area2lung = sum(nod_areas) / lung_area
            nod_num = len(datas)
        else:
            nod_area2lung = 0
            nod_size_max = 0
            nod_size_median = 0
            nod_size_min = 0
            nod_size_mean = 0
            nod_HU_max = None
            nod_HU_min = None
    #cavity
        hol_num = 0
        file_name = os.path.join(ele[0], date, LABEL_NAME[2])
        datas = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for e in reader:
                e[0] = e[0][2:]
                e[-1] = e[-1][:-2]
                datas.append(e)
        if datas:
            datas = np.array(datas, dtype=np.float)
            hol_areas = datas[:, 1]
            hol_HUs = datas[:, 2:3]
            real_areas = datas[:, 4]
            hol_size_max = max(hol_areas)
            hol_size_min = min(hol_areas)
            hol_size_mean = np.mean(hol_areas)
            hol_size_median = np.median(hol_areas)
            realhol2hol = real_areas / hol_areas
            realhol2hol_mean = np.mean(realhol2hol)
            hol_area2lung = sum(hol_areas) / lung_area
            hol_num = len(datas)
            if hol_num > 20:
                print()
        else:
            hol_area2lung = 0
            hol_size_max = 0
            hol_size_median = 0
            hol_size_min = 0
            hol_size_mean = 0
            realhol2hol_mean = None
            hol_HU_max = None
            hol_HU_min = None
        all2lung = 0
        patients_data.append([patient_id, date, exu_area2lung, nod_area2lung, hol_area2lung])
    # patients_data.append([patient_id, patient_data])
patients_data_arr = np.array(patients_data)
np.save('fetures_newLung9_sever.npy', patients_data_arr)

import os
import numpy as np
import pandas as pd
import csv
import datetime


def save_txt(name_list, txt_name):
    file = open(txt_name, 'w')
    # file = open('img_add1.txt', 'w')
    for fp in name_list:
        file.write(str(fp))
        file.write('\n')

    file.close()

root = 'LabCount'
date_root = 'data/id_date_v2_serious.csv'
LABEL_NAME = ['label_Exu.csv', 'label_Nod.csv', 'label_Cav.csv', 'lung_area.csv', 'cause_lung9.csv']

name_list = []
for path, dirs, files in os.walk(root):
    if dirs and len(dirs[0]) == 8:
        dir_list = []
        for dir in dirs:
            dir_list.append(dir)
        name_list.append([path, dir_list])
        # print(name_list)

patients_data = []
id_time = []
for step, ele in enumerate(name_list):
    patient_data = []
    patient_id = ele[0][9:]

    admission_str = '0'

    with open(date_root, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for e in reader:
            try:
                e[0]
            except:
                print('err')
            if e[0] == patient_id:
                print(e[0])
                print(patient_id)
                admission_str = e[1]
                before_time = int(e[2])
                sever_str = e[3]
                break

    if admission_str == '0':
        continue
    admission_date = datetime.date(int(admission_str[:4]), int(admission_str[4:6]), int(admission_str[6:]))
    if sever_str == '-1':
        serious_date = None
    else:
        serious_date = datetime.date(int(sever_str[:4]), int(sever_str[4:6]), int(sever_str[6:]))

    for date in ele[1]:
        print(date)
        scan_date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:])) - admission_date
        scan_date = scan_date.days + before_time
        if serious_date:
            serious_days = serious_date - datetime.date(int(date[:4]), int(date[4:6]), int(date[6:]))
            serious_time = serious_days.days
        else:
            serious_time = None
        if admission_str == '19990101':
            scan_date = before_time

        # lung
        lung_area = 0.0
        file_name = os.path.join(ele[0], date, LABEL_NAME[3])
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for e in reader:
                lung_area += float(e[2][2:-2])

    #exu
        exu_area2lung = 0.0
        file_name = os.path.join(ele[0], date, LABEL_NAME[0])
        datas = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for e in reader:
                e[0] = e[0][2:]
                e[-1] = e[-1][:-2]
                datas.append(e)
        if datas:
            datas = np.array(datas, dtype=np.float)
            exu_HUs1 = datas[:, 2]
            exu_HUs0 = datas[:, 3]
            exu_HU_max = np.max(exu_HUs1) / lung_area
            exu_HU_min = np.min(exu_HUs0) / lung_area
            exu_area2lung = np.sum(datas[:, 1]) / lung_area
            exu_num = len(datas)
            if exu_HU_max > 100:
                print()
        else:
            exu_area2lung = 0
            exu_size_max = 0
            exu_size_median = 0
            exu_size_min = 0
            exu_size_mean = 0
            exu_num = 0
            exu_HU_max = None
            exu_HU_min = None

    #nodule
        nod_num = 0
        file_name = os.path.join(ele[0], date, LABEL_NAME[1])
        datas = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for e in reader:
                e[0] = e[0][2:]
                e[-1] = e[-1][:-2]
                datas.append(e)
        if datas:
            datas = np.array(datas, dtype=np.float)
            nod_areas = datas[:, 1]
            nod_areas_list = nod_areas.tolist()
            max_area_id = nod_areas_list.index(max(nod_areas))
            nod_HUs = datas[:, 2:]
            nod_size_max = max(nod_areas) / lung_area
            nod_size_min = min(nod_areas) / lung_area
            nod_size_mean = np.mean(nod_areas) / lung_area
            nod_size_median = np.median(nod_areas) / lung_area
            nod_HU_max = nod_HUs[max_area_id, 0]
            nod_HU_min = nod_HUs[max_area_id, 1]
            nod_area2lung = sum(nod_areas) / lung_area
            nod_num = len(datas)
        else:
            nod_area2lung = 0
            nod_size_max = 0
            nod_size_median = 0
            nod_size_min = 0
            nod_size_mean = 0
            nod_HU_max = None
            nod_HU_min = None
    #cavity
        hol_num = 0
        file_name = os.path.join(ele[0], date, LABEL_NAME[2])
        datas = []
        with open(file_name, 'r') as f:
            reader = csv.reader(f)
            for e in reader:
                e[0] = e[0][2:]
                e[-1] = e[-1][:-2]
                datas.append(e)
        if datas:
            datas = np.array(datas, dtype=np.float)
            hol_areas = datas[:, 1]
            hol_HUs = datas[:, 2:3]
            real_areas = datas[:, 4]
            hol_size_max = max(hol_areas) / lung_area
            hol_size_min = min(hol_areas) / lung_area
            hol_size_mean = np.mean(hol_areas) / lung_area
            hol_size_median = np.median(hol_areas) / lung_area
            realhol2hol = real_areas / hol_areas
            realhol2hol_mean = np.mean(realhol2hol)
            hol_area2lung = sum(hol_areas) / lung_area
            hol_num = len(datas)
            if hol_num > 20:
                print()
        else:
            hol_area2lung = 0
            hol_size_max = 0
            hol_size_median = 0
            hol_size_min = 0
            hol_size_mean = 0
            realhol2hol_mean = None
            hol_HU_max = None
            hol_HU_min = None
        all2lung = 0
        patient_data.append([scan_date, serious_time, exu_area2lung, exu_HU_max, exu_HU_min,
                             nod_num, nod_area2lung, nod_size_max, nod_size_median, nod_size_min, nod_size_mean, nod_HU_max, nod_HU_min,
                             hol_num, hol_area2lung, hol_size_max, hol_size_median, hol_size_min, hol_size_mean, realhol2hol_mean])
        if serious_time and serious_time < 0:
            continue
        else:
            id_time.append([patient_id, scan_date, serious_time, exu_area2lung, lung_area, exu_num, nod_num, hol_num])

    patients_data.append([patient_id, patient_data])
patients_data_arr = np.array(patients_data)
save_txt(id_time, 'output/id_lungarea_lesNum.txt')
np.save('output/fetures.npy', patients_data_arr)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tools_gpr import *

# fit GPR
kernel = ConstantKernel(constant_value=.5, constant_value_bounds=(.1, 1e3)) * RBF(length_scale=1.5, \
                                                                                  length_scale_bounds=(1, 5))
noise = .75
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=noise ** 2)

gpr_ols = GaussianProcessRegressor(optimizer=trust_region_optimizer, alpha=0.1, n_restarts_optimizer=3)
data_root = 'output/data_13_08422_real_12f_IDlist.mat'

data = sio.loadmat(data_root)
t_n = data['t_n']
t_n = t_n.astype(np.float64)
x_s_n = data['x_s_n']
t_p = data['t_p']
x_s_p = data['x_s_p']

z_s_origin = data['x_score']
z_t = data['t']
label = data['y']
z_s = preprocessing.scale(z_s_origin, axis=0, with_mean=True, with_std=True, copy=True)

z_t_n = z_t[label == 0]
z_t_n = z_t_n.astype(np.float64)
z_s_n = z_s[label == 0]
z_t_p = z_t[label == 1]
z_s_p = z_s[label == 1]

train_X, test_X, train_y, test_y = train_test_split(t_n, x_s_n, test_size=0.3, random_state=1)
train_x_z, test_x_z, train_y_z, test_y_z = train_test_split(z_t_n, z_s_n, test_size=.3, random_state=1)
# train_x_z, test_x_z, train_y_z, test_y_z = train_x_z[train_x_z<8], test_x_z[test_x_z<8], \
#                                            train_y_z[train_x_z<8], test_y_z[test_x_z<8]
train_x_z, test_x_z, train_y_z, test_y_z = train_x_z[7<train_x_z], test_x_z[7<test_x_z], \
                                           train_y_z[7<train_x_z], test_y_z[7<test_x_z]
train_x_z, test_x_z, train_y_z, test_y_z = train_x_z[train_x_z<29], test_x_z[test_x_z<29], \
                                           train_y_z[train_x_z<29], test_y_z[test_x_z<29]
# train_x_z, test_x_z, train_y_z, test_y_z = train_x_z[train_x_z>28], test_x_z[test_x_z>28], \
#                                            train_y_z[train_x_z>28], test_y_z[test_x_z>28]

# z_t_p, z_s_p = z_t_p[z_t_p<8], z_s_p[z_t_p<8]
z_t_p, z_s_p = z_t_p[7<z_t_p], z_s_p[7<z_t_p]
z_t_p, z_s_p = z_t_p[z_t_p<29], z_s_p[z_t_p<29]
# z_t_p, z_s_p = z_t_p[z_t_p>28], z_s_p[z_t_p>28]


# cat 5% noise
index = train_y_z.argsort(axis=0)
num_ct = len(index)

plt_test_x = np.arange(1, 100, .1).reshape(-1, 1)


train_x_z = train_x_z.reshape(-1, 1)
test_x_z = test_x_z.reshape(-1, 1)
train_y_z = train_y_z.ravel()
gpr.fit(train_x_z, train_y_z)
gpr_ols.fit(train_x_z, train_y_z)
mu, cov = gpr.predict(plt_test_x, return_cov=True)
mu_ols, cov_ols = gpr_ols.predict(plt_test_x, return_cov=True)
plt_test_y = mu.ravel()
plt_test_y_ols = mu_ols.ravel()
uncertainty = 3 * np.sqrt(np.diag(cov))  # 1.96 99.7%
uncertainty_in = 1.96 * np.sqrt(np.diag(cov))  # 1.96   95%c IC
print(max(uncertainty), max(uncertainty))
# plotting
plt.figure()
plt.xlim([0, 100])

## GPR
plt.scatter(train_x_z, train_y_z, label="training data", c="#CACACA", marker="o")  # 标准化数据
plt.scatter(test_x_z, test_y_z, label="test data", c="#A1C7E0", marker="o")
plt.scatter(z_t_p, z_s_p, label="severed data", c="#E3C75F", marker="*")


print('Train acc:{:.2f}'.format(gpr.score(train_X, train_y)))
print('Test acc:{:.2f}'.format(gpr.score(test_X, test_y)))
plt.xlabel('Time from the onset of initial symptoms(d)')
plt.ylabel('CT Score')
plt.legend(loc='best')
plt.show()

# norm_ctscore
plt.scatter(train_x_z, train_y_z, label="training data", c="#CACACA", marker="o")
plt.scatter(test_x_z, test_y_z, label="test data", c="#A1C7E0", marker="o")
plt.scatter(z_t_p, z_s_p, label="severed data", c="#E3C75F", marker="*")
plt.plot([0, 100], [0, 0], color="black", lw=.5, linestyle="--", alpha=0.5)
plt.xlabel('Time from the onset of initial symptoms(d)')
plt.ylabel('Z-score')
plt.legend(loc='best')
plt.show()

merics_roc_v2(train_x_z, train_y_z, test_x_z, test_y_z, z_t_p, z_s_p,
              [plt_test_x, plt_test_y, uncertainty_in, uncertainty], is_best=True)
merics_roc_v2(train_x_z, train_y_z, test_x_z, test_y_z, z_t_p, z_s_p,
              [plt_test_x, plt_test_y_ols, uncertainty_in, uncertainty])

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

import csv
import codecs

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
            file_csv = codecs.open(file_name,'w+','utf-8')#追加
            writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            # for data in datas:
            #     writer.writerow(data)
            writer.writerows(map(lambda x: [x], datas))
            print("Written done!")
import numpy as np
import cv2
from skimage import measure, color
import save_csv
import copy
import os
from natsort import natsorted
import nibabel as nib
import torch

from scipy.io import savemat
import scipy.io as sio

id_name = ['Exu', 'Nod', 'Cav']

     
def noise_contorller(img_arr):
    # median blur [5, 5]
    slice_num = len(img_arr)
    for i in range(slice_num):
        img_arr[i] = cv2.medianBlur(img_arr[i], 5)
    return img_arr

def label_imgs(imgs, labels):
    labels_mask = np.zeros([3, len(labels), imgs.shape[1], imgs.shape[2]])
    imgs_mask = np.zeros([3, len(imgs), imgs.shape[1], imgs.shape[2]])
    for i in range(1, 4):
        print()
        label_temp = copy.deepcopy(labels)
        label_temp[label_temp != i] = 0
        label_temp[label_temp == i] = 1
        labels_mask[i-1] = label_temp
        imgs_mask[i-1] = label_temp * imgs
    return labels_mask, imgs_mask

def compute_region(labels_mask, lungs_seg, imgs_mask,  metadata, save_path):
    voxes_v = metadata[0] * metadata[1] * metadata[2]
    label_num = len(labels_mask)

    # lungSeg
    lung_lab = measure.label(lungs_seg, connectivity=2)
    lung_properties = measure.regionprops(lung_lab)
    lung_area = []
    for lung_prop in lung_properties:
        if lung_prop.area>1e3:
            lung_area.append([voxes_v, lung_prop.area, lung_prop.area * voxes_v])
    save_csv.data_write_csv(save_path + '/lung_area.csv', lung_area)           # save the V of lung [pixels, V_meta, real_V]

    # lesion features
    for i in range(label_num):
        label_mask = labels_mask[i] * lungs_seg

        con_lab = measure.label(label_mask, connectivity=2)
        properties = measure.regionprops(con_lab, imgs_mask)

        valid_label = []
        for prop in properties:
            if prop.area>10:
                inten_image = copy.deepcopy(prop.intensity_image)
                if i==0:
                    inten_image[inten_image>600]= -9999
                if i==1:
                    inten_image[inten_image>100]= -9999
                if i==2:
                    real_hole_area = np.sum(inten_image<0)
                inten_image = np.reshape(inten_image, [inten_image.size])
                inten_image = np.sort(inten_image[inten_image>-9999])

                max_HU = inten_image[round(inten_image.size * 0.95) - 1 ]
                min_HU = inten_image[round(inten_image.size * 0.05) - 1 ]

                if i==2:
                    valid_label.append([prop.label, prop.area * voxes_v, max_HU, min_HU, real_hole_area * voxes_v])
                else:
                    valid_label.append([prop.label, prop.area * voxes_v, max_HU, min_HU])


        save_csv.data_write_csv(save_path + f'/label_{id_name[i]}.csv', valid_label)  # save lesions V 2 Lung
        print('regions number:', np.max(con_lab))


def LabCount(data_list):
    for i, names in enumerate(data_list):
        if not names[1]:
            continue
        save_root = './output/LabCount/' + names[0][12:]
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        pices_num = len(names[1])
        labels = np.zeros([pices_num, 512, 512])
        nii_root = names[0].replace('CT_img', 'CT_data')

        nii_root = nii_root + '.nii'
        img_ct = nib.load(nii_root)
        img_arr = img_ct.get_fdata()
        header = img_ct.header
        metadata = header.get_zooms()

        if len(img_arr.shape)==4:
            img_arr = img_arr[:, :, :, 0]
        img_arr = np.array(img_arr, np.int16)
        img_arr = np.transpose(img_arr, (2, 1, 0))
        img_arr = noise_contorller(img_arr)

        lungs_seg = np.zeros([pices_num, img_arr.shape[1], img_arr.shape[2]])
        for k in range(pices_num):
            name = os.path.join(names[0], names[1][k])
            lab_name = name.replace('CT_img', 'lesions')
            lung_root = name.replace('CT_img/', 'lung_seg/')
            lung_root = lung_root.replace('.npy', '.mat')

            label = np.load(lab_name)
            labels[k, :, :] = label

            lung_seg = sio.loadmat(lung_root)
            lung_seg = lung_seg['lung_prob']

            lung_seg_arr = np.array(lung_seg)
            lungs_seg[k,:,:] = lung_seg_arr

        labels_o = torch.from_numpy(labels)
        labels_o = labels_o.unsqueeze(0)
        seg = torch.nn.functional.interpolate(labels_o, size=img_arr.shape[1:], mode='bilinear')
        labels = seg.numpy()[0]

        labels_mask, imgs_mask = label_imgs(img_arr, labels)
        compute_region(labels_mask, lungs_seg, img_arr, metadata, save_root)

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    data_list = []
    lung_lab_list = []
    for root, dirs, files in os.walk('data/CT_img/'):
        names = []
        for name in files:
            if 'npy' in name:
                names.append(name)
        names = natsorted(names)
        data_list.append([root, names])
    data_list = natsorted(data_list)
    LabCount(data_list)

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from PIL import Image
from skimage import measure, color
from scipy import ndimage
import cv2

import scipy.io as scio

def lung_9mask_old(lungs_seg):
        lungs_prespective = np.sum(lungs_seg, 1)
        lungs_prespective = np.array(lungs_prespective, dtype= bool)
        save_np = Image.fromarray(lungs_prespective)
        save_np.save('lungs_prespective.png')
        slience = lungs_prespective.shape[0]
        top_slience = 0
        down_slience = slience
        left_edge = []
        right_edge = []
        for i in range(slience):
            if top_slience == 0 and np.sum(lungs_prespective[i]):
                top_slience = i
            if top_slience != 0 and np.sum(lungs_prespective[i])==0:
                down_slience = 0
            if np.sum(lungs_prespective[i]):
                for j in range(lungs_prespective.shape[1]):
                    if lungs_prespective[i, j]:
                        if left_edge and left_edge[-1][0] != i - 1:
                            if len(left_edge)>10:
                                break
                            else:
                                left_edge = []

                        left_edge.append([i, j])
                        break
                for j in range(lungs_prespective.shape[1]):
                    if lungs_prespective[i, lungs_prespective.shape[1] - j - 1]:
                        if right_edge and right_edge[-1][0] != i - 1:
                            if len(right_edge)>10:
                                break
                            else:
                                right_edge = []
                        right_edge.append([i, lungs_prespective.shape[1] - j])
                        break
        
        left_edge, right_edge = np.array(left_edge), np.array(right_edge)
        left_outp = np.argmin(left_edge[:,1])
        right_outp = np.argmax(right_edge[:, 1])
        left_edge, right_edge = left_edge[left_outp:], right_edge[right_outp:]
        f_left = np.polyfit(left_edge[:, 0], left_edge[:, 1], 2)
        f_l = np.poly1d(f_left)
        y_l = f_l(left_edge[:, 0])
        # plt.plot(left_edge[:, 0], y1, label='fit val')
        f_right = np.polyfit(right_edge[:, 0], right_edge[:, 1], 2)
        f_r = np.poly1d(f_right)
        y_r = f_r(right_edge[:, 0])
        # plt.plot(left_edge[:, 0], left_edge[:, 1])
        # plt.plot(right_edge[:, 0], right_edge[:, 1])    
        # plt.savefig('lung_edgefit.png')  
        top_1th = int(round(left_edge[0,0] + (left_edge[-1,0] - left_edge[0,0]) / 3))
        top_2th = int(round(right_edge[0,0] + (right_edge[-1,0] - right_edge[0,0]) *2/ 3))
        left_edge1th = copy.deepcopy(left_edge)
        right_edge1th = copy.deepcopy(right_edge)
        left_edge2th = copy.deepcopy(left_edge)
        right_edge2th = copy.deepcopy(right_edge)

        for i in range(lungs_prespective.shape[1]):
            if lungs_prespective[0, i] == 1 and lungs_prespective[top_1th, i+1] == 0:
                w_l = int(sum(lungs_prespective[top_1th, :i]) / 3)
                left_edge1th[:, 1] = y_l + w_l
                left_edge2th[:, 1] = y_l + 2*w_l
                w_r = int(sum(lungs_prespective[top_1th, i+1:]) / 3)
                right_edge1th[:, 1] = y_r - w_r
                right_edge2th[:, 1] = y_r - 2*w_r
                break
            if lungs_prespective[top_2th, i] == 1 and lungs_prespective[top_2th, i+1] == 0:
                w_l = int(sum(lungs_prespective[top_2th, :i]) / 3)
                left_edge1th[:, 1] = y_l + w_l
                left_edge2th[:, 1] = y_l + 2*w_l
                w_r = int(sum(lungs_prespective[top_2th, i+1:]) / 3)
                right_edge1th[:, 1] = y_r - w_r
                right_edge2th[:, 1] = y_r - 2*w_r
                break
        plt.plot(right_edge[:, 0], right_edge[:, 1]) 
        plt.plot(right_edge1th[:, 0], right_edge1th[:, 1]) 
        plt.plot(right_edge2th[:, 0], right_edge2th[:, 1])  
        plt.plot(left_edge[:, 0], left_edge[:, 1]) 
        plt.plot(left_edge1th[:, 0], left_edge1th[:, 1]) 
        plt.plot(left_edge2th[:, 0], left_edge2th[:, 1]) 
        plt.savefig('lung_edgefit.png')  


def fun(x):
    round(x, 2)
    if x>=0: return '+'+str(x)
    else: return str(x)

def fit_surface(X, Y, Z):
    n = len(X)
    sigma_x = 0
    for i in X : sigma_x += i
    sigma_y = 0
    for i in Y: sigma_y += i
    sigma_z = 0
    for i in Z: sigma_z += i
    sigma_x2 = 0
    for i in X: sigma_x2 += i * i
    sigma_y2 = 0
    for i in Y: sigma_y2 += i * i
    sigma_x3 = 0
    for i in X: sigma_x3 += i * i * i
    sigma_y3 = 0
    for i in Y: sigma_y3 += i * i * i
    sigma_x4 = 0
    for i in X: sigma_x4 += i * i * i * i
    sigma_y4 = 0
    for i in Y: sigma_y4 += i * i * i * i
    sigma_x_y = 0
    for i in range(n):
        sigma_x_y += X[i] * Y[i]
    # print(sigma_xy)
    sigma_x_y2 = 0
    for i in range(n): sigma_x_y2 += X[i] * Y[i] * Y[i]
    sigma_x_y3 = 0
    for i in range(n): sigma_x_y3 += X[i] * Y[i] * Y[i] * Y[i]
    sigma_x2_y = 0
    for i in range(n): sigma_x2_y += X[i] * X[i] * Y[i]
    sigma_x2_y2 = 0
    for i in range(n): sigma_x2_y2 += X[i] * X[i] * Y[i] * Y[i]
    sigma_x3_y = 0
    for i in range(n): sigma_x3_y += X[i] * X[i] * X[i] * Y[i]
    sigma_z_x2 = 0
    for i in range(n): sigma_z_x2 += Z[i] * X[i] * X[i]
    sigma_z_y2 = 0
    for i in range(n): sigma_z_y2 += Z[i] * Y[i] * Y[i]
    sigma_z_x_y = 0
    for i in range(n): sigma_z_x_y += Z[i] * X[i] * Y[i]
    sigma_z_x = 0
    for i in range(n): sigma_z_x += Z[i] * X[i]
    sigma_z_y = 0
    for i in range(n): sigma_z_y += Z[i] * Y[i]
    # print("-----------------------")
    # 给出对应方程的矩阵形式
    a = np.array([[sigma_x4, sigma_x3_y, sigma_x2_y2, sigma_x3, sigma_x2_y, sigma_x2],
                  [sigma_x3_y, sigma_x2_y2, sigma_x_y3, sigma_x2_y, sigma_x_y2, sigma_x_y],
                  [sigma_x2_y2, sigma_x_y3, sigma_y4, sigma_x_y2, sigma_y3, sigma_y2],
                  [sigma_x3, sigma_x2_y, sigma_x_y2, sigma_x2, sigma_x_y, sigma_x],
                  [sigma_x2_y, sigma_x_y2, sigma_y3, sigma_x_y, sigma_y2, sigma_y],
                  [sigma_x2, sigma_x_y, sigma_y2, sigma_x, sigma_y, n]])
    b = np.array([sigma_z_x2, sigma_z_x_y, sigma_z_y2, sigma_z_x, sigma_z_y, sigma_z])
    # 高斯消元解线性方程
    res = np.linalg.solve(a, b)

    return res


def lung_9mask(lung9, edges):
    # fig = plt.figure()  # 建立一个空间
    # ax = fig.add_subplot(111, projection='3d')  # 3D坐标
    res_equation = []
    for i in range(2):
        # i = 1
        edge = edges[i]


        edge1th = np.array(edge[0])
        edge2th = np.array(edge[1])
        edge3th = np.array(edge[2])

        X, Y, Z = edge1th[:, 0], edge1th[:, 1], edge1th[:, 2]
        X1, Y1, Z1 = edge2th[:, 0], edge2th[:, 1], edge2th[:, 2]
        X2, Y2, Z2 = edge3th[:, 0], edge3th[:, 1], edge3th[:, 2]

        res = fit_surface(X, Y, Z)
        res1 = fit_surface(X1, Y1, Z1)
        res2 = fit_surface(X2, Y2, Z2)
        res_equation.append([res, res1, res2])
        print("z=%.6s*x^2%.6s*xy%.6s*y^2%.6s*x%.6s*y%.6s" % (
            fun(res[0]), fun(res[1]), fun(res[2]), fun(res[3]), fun(res[4]), fun(res[5])))
        x_start, x_end = min(X), max(X)
        y_start, y_end = min(Y), max(Y)
        x_1th, x_2th = round((x_end - x_start) / 3 + x_start), round(2 * (x_end - x_start) / 3 + x_start)
        ux = np.linspace(x_start, x_end, x_end - x_start + 1)  # 创建一个等差数列
        uy = np.linspace(y_start, y_end, y_end - y_start + 1)  # 创建一个等差数列
        x, y = np.meshgrid(ux, uy)

        z = res[0] * x * x + res[1] * x * y + res[2] * y * y + res[3] * x + res[4] * y + res[5]
        z1 = res1[0] * x * x + res1[1] * x * y + res1[2] * y * y + res1[3] * x + res1[4] * y + res1[5]
        z2 = res2[0] * x * x + res2[1] * x * y + res2[2] * y * y + res2[3] * x + res2[4] * y + res2[5]
        x_all, y_all, z_all, z1_all, z2_all = x.flatten('F').astype(int), (y.flatten('F')).astype(int), \
                                              np.round(z.flatten('F')).astype(int), np.round(z1.flatten('F')).astype(int), \
                                              np.round(z2.flatten('F')).astype(int)
        # plt.scatter(y_all, z_all, s=1)
        # plt.scatter(y_all, z1_all, s=1)
        # plt.scatter(y_all, z2_all, s=1)
        # plt.show()
        z_all[z_all<0] = 0
        z1_all[z1_all<0] = 0
        z2_all[z2_all<0] = 0
        mid_line = int(lung9.shape[3]/2)
        # left
        if i == 0:
            for id, x_i in enumerate(x_all):
                if x_i < x_1th:
                    lung9[0, x_i, y_all[id], z_all[id]:z1_all[id]] = 1
                    lung9[1, x_i, y_all[id], z1_all[id]:z2_all[id]] = 1
                    lung9[2, x_i, y_all[id], z2_all[id]:mid_line] = 1
                elif x_i < x_2th:
                    lung9[3, x_i, y_all[id], z_all[id]:z1_all[id]] = 1
                    lung9[4, x_i, y_all[id], z1_all[id]:z2_all[id]] = 1
                    lung9[5, x_i, y_all[id], z2_all[id]:mid_line] = 1
                else:
                    lung9[6, x_i, y_all[id], z_all[id]:z1_all[id]] = 1
                    lung9[7, x_i, y_all[id], z1_all[id]:z2_all[id]] = 1
                    lung9[8, x_i, y_all[id], z2_all[id]:mid_line] = 1

        # for j in range(11, 34):
        #     img_arr = np.zeros_like(lung9[0, 0, :, :])
        #     for i in range(9):
        #         img_arr = lung9[i, j, :, :] * 255 * (1+i) / 9 + img_arr
        #     img = Image.fromarray(img_arr)
        #     # img.save(f'./res_lung9/left_{i}.jpg')
        #     img.convert('RGB').save(f'./res_lung9/left_{j}.jpg')
        # right
        if i == 1:
            for id, x_i in enumerate(x_all):
                if x_i < x_1th:
                    lung9[9, x_i, y_all[id], z1_all[id]:z_all[id]] = 1
                    lung9[9 + 1, x_i, y_all[id], z2_all[id]:z1_all[id]] = 1
                    lung9[9 + 2, x_i, y_all[id], mid_line:z2_all[id]] = 1
                elif x_i < x_2th:
                    lung9[9 + 3, x_i, y_all[id], z1_all[id]:z_all[id]] = 1
                    lung9[9 + 4, x_i, y_all[id], z2_all[id]:z1_all[id]] = 1
                    lung9[9 + 5, x_i, y_all[id], mid_line:z2_all[id]] = 1
                else:
                    lung9[9 + 6, x_i, y_all[id], z1_all[id]:z_all[id]] = 1
                    lung9[9 + 7, x_i, y_all[id], z2_all[id]:z1_all[id]] = 1
                    lung9[9 + 8, x_i, y_all[id], mid_line:z2_all[id]] = 1
    return lung9, res_equation

def features_lung9(labels_mask, lung9, lungs_seg):
    features2lung9 = []
    label_num = len(labels_mask)
    for i in range(label_num):
        label_mask = labels_mask[i]
        feature2lung9 = []
        for i in range(18):
            casue_lung9 = lung9[i] * lungs_seg * label_mask
            lung_lab = measure.label(casue_lung9, connectivity=2)
            feature2lung9.append(np.max(lung_lab))
        features2lung9.append(feature2lung9)
    return features2lung9 #, lung9   ????210717


def imfill(img):
    output = ndimage.binary_fill_holes(img).astype(bool)
    return output

def closeopration(img):
    kernel = np.ones((5, 5), np.uint8)
    iclose = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return iclose


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.optimize import least_squares, leastsq


def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()


def line_total_least_squares(x_in, y_in):
    n = len(x_in)

    x_m = np.sum(x_in) / n
    y_m = np.sum(y_in) / n

    # Calculate the x~ and y~
    x1 = x_in - x_m
    y1 = y_in - y_m

    # Create the matrix array
    X = np.vstack((x1, y1))
    X_t = np.transpose(X)

    # Finding A_T_A and it's Find smallest eigenvalue::
    prd = np.dot(X, X_t)
    W, V = np.linalg.eig(prd)
    small_eig_index = W.argmin()
    a, b = V[:, small_eig_index]

    # Compute C:
    c = (-1 * a * x_m) + (-1 * b * y_m)

    return a, b, c


def merics_roc(test_x_z, test_y_z, z_t_p, z_s_p):
    # [4.7, -1.5]
    cur_k = 4.7
    fps, tps = [], []
    for i in range(62):
        cur_k -= .1
        fp = np.sum(test_y_z > cur_k)
        fn = np.sum(z_s_p < cur_k)
        tp = np.sum(z_s_p > cur_k)
        tn = np.sum(test_y_z < cur_k)
        fps.append(fp)
        tps.append(tp)
    num = len(test_y_z) + len(z_s_p)
    fpr = np.array(fps) / len(z_s_p)
    tpr = np.array(tps) / len(z_s_p)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="#196774",
        lw=lw,
        label="L-BFGS: AUC=%0.2f" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("$N_{test}$=39, $N_{severed}$=39")
    plt.legend(loc="lower right")
    # plt.show()


def merics_roc_v2(train_x_z, train_y_z, test_x_z, test_y_z, z_t_p, z_s_p, plt_test, is_best=False, uncertain_in=None):
    plt_x, plt_y = plt_test[0], plt_test[1]
    uncertain_in, uncertain = plt_test[2], plt_test[2]
    plt_y = plt_y.reshape([-1, 1])
    for i in range(len(test_x_z)):
        index = test_x_z[i]
        index_plt = int(index - 1) * 10
        test_y_z[i] -= plt_y[index_plt]
        index = z_t_p[i]
        index_plt = int(index - 1) * 10
        z_s_p[i] -= plt_y[index_plt]
    for i in range(len(train_x_z)):
        index = train_x_z[i]
        if index > 98:
            continue
        index_plt = int(index - 1) * 10
        train_y_z[i] -= plt_y[index_plt]
    if is_best:
        # test_y_z = abs(test_y_z)
        # z_s_p = abs(z_s_p)
        plt.scatter(train_x_z, train_y_z, label="training data", c="#CACACA", marker="o")
        plt.scatter(test_x_z, test_y_z, label="test data", c="#A1C7E0", marker="o")
        plt.scatter(z_t_p, z_s_p, label="severed data", c="#E3C75F", marker="*")
        plt.plot([0, 100], [0, 0], color="black", lw=.5, linestyle="--", alpha=0.5)
        plt.xlabel('Time from the onset of initial symptoms(d)')
        plt.ylabel('Z-score')
        plt.legend(loc='best')
        # plt.savefig('../figs/z-score.pdf')
        plt.show()
    # ROC
    # [2.1 -0.6]; 标准化后：[-0.5, 4.7]; I:[-0.289, 2.6]
    cur_k = max(max(test_y_z), max(z_s_p))
    fps, tps = [], []
    # V1 maybe wrong
    #     for i in range(520):
    #         cur_k -= .01
    #         # cur_k = 0
    #         fp = np.sum(test_y_z > cur_k)
    #         fn = np.sum(z_s_p < cur_k)
    #         tp = np.sum(z_s_p > cur_k)
    #         tn = np.sum(test_y_z < cur_k)
    #         fps.append(fp)
    #         tps.append(tp)
    #     num = len(test_y_z) + len(z_s_p)
    #     fpr = np.array(fps) / len(z_s_p)
    #     tpr = np.array(tps) / len(z_s_p)
    # V2
    while (min(min(test_y_z), min(z_s_p)) < cur_k):
        # for i in range(600):
        cur_k -= .01
        # cur_k = 0
        # test_y_z = abs(test_y_z)
        # z_s_p = abs(z_s_p)

        fp = np.sum(test_y_z > cur_k)
        fn = np.sum(z_s_p < cur_k)
        tp = np.sum(z_s_p > cur_k)
        tn = np.sum(test_y_z < cur_k)
        fps.append(fp)
        tps.append(tp)
    num = len(test_y_z) + len(z_s_p)
    fpr = np.array(fps) / len(test_y_z)
    tpr = np.array(tps) / len(z_s_p)

    roc_auc = auc(fpr, tpr)

    # plt.figure()
    lw = 2
    if is_best:
        plt.plot(
            fpr,
            tpr,
            color="#196774",
            lw=lw,
            label="L-BFGS: AUC=%0.2f" % roc_auc,
        )
        np.savez('roc_best_s2.npz', fpr=fpr,tpr=tpr)
    else:
        plt.plot(
            fpr,
            tpr,
            color="#196774",
            linestyle=':',
            lw=lw,
            label="    OLS: AUC=%0.2f" % roc_auc,
        )
        np.savez('roc_OLS_s2.npz', fpr=fpr, tpr=tpr)
        plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("$N_{test}$=%s, $N_{severed}$=%s" % (len(test_y_z), len(z_s_p)))
        plt.legend(loc="lower right")
        # plt.savefig('../figs/ROC_stage3.pdf')
        plt.show()


def plot_id_list(ids_list, date, label, score):
    for i in range(99):
        patient_id = i + 1
        date_p = date[label == 1]
        score_p = score[label == 1]
        p_x = date_p[ids_list[label == 1] == patient_id]
        p_y = score_p[ids_list[label == 1] == patient_id]
        plt.plot(p_x, p_y, c='#E3C75F', alpha=0.3)

        date_n = date[label == 0]
        score_n = score[label == 0]
        p_x = date_n[ids_list[label == 0] == patient_id]
        p_y = score_n[ids_list[label == 0] == patient_id]
        plt.plot(p_x, p_y, c='#CACACA', alpha=0.3)


def trust_region_optimizer(obj_func, initial_theta, bounds):
    trust_region_method = leastsq(1 / obj_func, initial_theta, bounds, method='trf')
    return (trust_region_method.x, trust_region_method.fun)
