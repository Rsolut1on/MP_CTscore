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
