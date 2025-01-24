#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:27:03 2023

@author: liyuan
"""

import numpy as np
import nibabel as nib
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import  color
import os
import SimpleITK as sitk
def hessian_enhance(input_file, output_file):
    # nii_image = nib.load(input_file)
    # src_image = nii_image.get_fdata()
    src_image = sitk.ReadImage(input_file)
    src_arr = sitk.GetArrayFromImage(src_image)

    depth = src_arr.shape[2]
    out_image = np.zeros(src_arr.shape, dtype=np.float32)
    img = []

    for z in range(depth):
        image=src_arr[:,:,z]
        # print(max(image.reshape(-1,)))
        # print(min(image.reshape(-1,)))

        if len(image.shape) == 3:
            image = color.rgb2gray(image)

        # Compute the Hessian matrix
        hessian_matrix_result = hessian_matrix(image, sigma=0.1)

        # Compute the eigenvalues of the Hessian matrix
        eigenvalues = hessian_matrix_eigvals(hessian_matrix_result)
        # vesselness = np.zeros_like(image)
        # mask = eigenvalues[0] < 0
        # vesselness[mask] = np.abs(eigenvalues[0])[mask]        # print(eigenvalues[1].shape)
        out_image[:,:,z] = eigenvalues[0]


    # out_nii = nib.Nifti1Image(out_image, nii_image.affine)
    out_nii = sitk.GetImageFromArray(out_image)
    out_nii.SetOrigin(src_image.GetOrigin())
    out_nii.SetDirection(src_image.GetDirection())
    out_nii.SetSpacing(src_image.GetSpacing())
    print(f"----------------------------{input_file}")
    print(max(out_image.reshape(-1,)))
    print(min(out_image.reshape(-1,)))
    # nib.save(out_nii, output_file)
    sitk.WriteImage(out_nii,output_file)
    return 0

    
path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/semi_imagesTr_ori/"
save_path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/semi_full_imagesTr_hessian/"
os.makedirs(save_path, exist_ok=True)

def main():
    for nii_data in os.listdir(path):
        # if "CXM"  in nii_data:
        # try:
            print(f"processing {nii_data}")
            # new_name = nii_data.split(".nii.gz")[0] + "_hessian_0000.nii.gz"
            # if new_name in os.listdir(save_path):
            #     continue
            # res = hessian_enhance(path +nii_data, output_file=save_path+new_name)
            if nii_data in os.listdir(save_path):
                continue
            else:
                res = hessian_enhance(path +nii_data, output_file=save_path+nii_data)

        # except:
            
        #     print(f"error in {nii_data}")
        #     continue
if __name__ == "__main__":
        
    main()