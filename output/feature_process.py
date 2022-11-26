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
