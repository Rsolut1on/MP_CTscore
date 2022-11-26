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

