#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
import itertools
import os
import copy 
def count_consecutive_same_values(input_list):
    return [len(list(group)) for key, group in itertools.groupby(input_list)]

# Read the CT image and the right lung label
def compute_mask( label,save_ct,save_label):
    # ct_file_name = ct
    # ct_image = sitk.ReadImage(ct_file_name)

    label_file_name = label
    label_image = sitk.ReadImage(label_file_name)

    # Convert the ITK images to NumPy arrays
    # ct_array = sitk.GetArrayFromImage(ct_image)
    label_array = sitk.GetArrayFromImage(label_image)
    # print(label_array.shape,ct_array.shape)

    # sum_row = label_array.sum(axis=0).sum(axis=0)
    # binary_array = sum_row > 0
    # counter = count_consecutive_same_values(binary_array)
    # print(label_file_name)
    # print("counter:", counter)
    # assert len(counter) == 3
    
    # mx_counter = max(counter)
    # right_lung = 1
    # # if np.sum(np.logical_not(np.array(counter) >= 256)) >= 1:
    # if counter[0] == mx_counter: #right lung
    #     # mask = [3] * 256 + [0] * 256
    #     right_lung = 1
    # elif counter[-1] == mx_counter: #left lung
    #     # mask = [0] * 256 + [3] * 256
    #     right_lung = 0
    # # else:
    # #     mask = [0] * 512
    # # mask = np.tile(mask, (512,1))

    # # full_mask =  np.tile(mask, ( label_array.shape[0],1, 1))
    Depth = label_array.shape[0]
    right_lung = copy.deepcopy(label_array)#[:, :, 256:]
    left_lung = copy.deepcopy(label_array)#[:, :, :256]
    right_lung[:,:,:256] = 0
    left_lung[:,:,256:] = 0

    save_label_left = save_label.split(".nii.gz")[0]+"_left.nii.gz"
    save_label_right = save_label.split(".nii.gz")[0]+"_right.nii.gz"
    print(f"right_lung shape: {right_lung.shape},left_lung shape: {left_lung.shape},label shape: {label_array.shape}")
    print(f"right_lung unique: {np.unique(right_lung)},left_lung unique: {np.unique(left_lung)},label unique: {np.unique(label_array)}")
    savemask = sitk.GetImageFromArray(right_lung)
    savemask.SetOrigin(label_image.GetOrigin())
    savemask.SetDirection(label_image.GetDirection())
    savemask.SetSpacing(label_image.GetSpacing())
    sitk.WriteImage(savemask,save_label_right)

    savemask = sitk.GetImageFromArray(left_lung)
    savemask.SetOrigin(label_image.GetOrigin())
    savemask.SetDirection(label_image.GetDirection())
    savemask.SetSpacing(label_image.GetSpacing())
    print(f"savemask shape: {savemask.GetSize()}")
    sitk.WriteImage(savemask,save_label_left)
    # sitk.WriteImage(saveimg, save_ct)
    
    return 0

ct_path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/imagesTr_ori"
lb_path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/labelsTr_ori"


# save_ct = r"/pub/data/yangdeq/air_seg/vein/nnUNet_raw/Dataset009_vein_semi_seg_cut_half/imagesTr"
save_label = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/labelsTr_ori_split"


# os.makedirs(save_ct, exist_ok=True)
# os.makedirs(save_label, exist_ok=True)
# os.makedirs(save_images, exist_ok=True)

for i in os.listdir(lb_path):
        compute_mask( \
            os.path.join(lb_path,i),os.path.join(i.split(".nii.gz")[-2]+"_0000.nii.gz"),os.path.join(save_label,i))
        # break

