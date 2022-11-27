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
np.save('fetures.npy', patients_data_arr)
