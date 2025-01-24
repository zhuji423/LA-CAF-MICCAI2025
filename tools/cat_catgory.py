import SimpleITK as sitk
import os
import numpy as np
path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/semi_full_labelsTr_ori"
for file in os.listdir(path):
    # if file.endswith(".nii.gz"):
        img = sitk.ReadImage(os.path.join(path, file))
        arr = sitk.GetArrayFromImage(img)
        print(file, arr.shape, np.unique(arr))
        # print(file, img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection())
        # break

