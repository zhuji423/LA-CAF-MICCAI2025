#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
import itertools
import os
import copy 
def count_consecutive_same_values(input_list):
    return [len(list(group)) for key, group in itertools.groupby(input_list)]
### 动脉1，静脉2   
# 改右肺动脉3，静脉4 
# Read the CT image and the right lung label
def compute_mask( label,save_ct,save_label):
    # ct_file_name = ct
    # ct_image = sitk.ReadImage(ct_file_name)

    label_file_name = label
    label_image = sitk.ReadImage(label_file_name)
    label_array = sitk.GetArrayFromImage(label_image)

    Depth = label_array.shape[0]
    right_lung = copy.deepcopy(label_array)#[:, :, 256:]
    left_lung = copy.deepcopy(label_array)#[:, :, :256]
    right_lung[:,:,:256] = 0
    left_lung[:,:,256:] = 0

    right_lung[right_lung == 1] = 3
    right_lung[right_lung == 2] = 4

    merge_label = right_lung + left_lung
    
    print(f"right_lung shape: {right_lung.shape},left_lung shape: {left_lung.shape},label shape: {label_array.shape}")
    print(f"right_lung unique: {np.unique(right_lung)},left_lung unique: {np.unique(left_lung)},label unique: {np.unique(merge_label)}")

    savemask = sitk.GetImageFromArray(merge_label)
    savemask.SetOrigin(label_image.GetOrigin())
    savemask.SetDirection(label_image.GetDirection())
    savemask.SetSpacing(label_image.GetSpacing())
    print(f"savemask shape: {savemask.GetSize()}")
    sitk.WriteImage(savemask,save_label)
    # sitk.WriteImage(saveimg, save_ct)
    
    return 0

ct_path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/imagesTr_ori"
lb_path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/semi_labelsTr_ori"
# lb_path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/labelsTr_ori"


# save_ct = r"/pub/data/yangdeq/air_seg/vein/nnUNet_raw/Dataset009_vein_semi_seg_cut_half/imagesTr"
save_label = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/semi_full_labelsTr_ori"


# os.makedirs(save_ct, exist_ok=True)
# os.makedirs(save_label, exist_ok=True)
# os.makedirs(save_images, exist_ok=True)

for i in os.listdir(lb_path):
        if not os.path.exists(os.path.join(save_label,i)):
            compute_mask( \
                os.path.join(lb_path,i),os.path.join(i.split(".nii.gz")[-2]+"_0000.nii.gz"),os.path.join(save_label,i))
        # break
        else:
            print(f"{i} exists")
            continue

