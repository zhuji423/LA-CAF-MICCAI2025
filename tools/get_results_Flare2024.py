import os 
import SimpleITK as sitk
import numpy as np
path = r"/pub/data/yangdeq/Flare2024/validation_infer_clip_unet"
save_path = r"/pub/data/yangdeq/Flare2024/validation_infer_clip_unet/save_results"
for names in os.listdir(path):
    save_array = None
    print(names+".nii.gz")
    if names.split("_0000")[0]+".nii.gz" not in os.listdir(save_path):
        for  nii_files in os.listdir(os.path.join(path, names)):
            if "Tumor" in nii_files or "Cyst" in nii_files:
                ## read the nii file
                img = sitk.ReadImage(os.path.join(path, names, nii_files))
                arr = sitk.GetArrayFromImage(img)
                print(arr.shape,nii_files,np.unique(arr))
                if save_array is None:
                    save_array = arr
                else:
                    save_array += arr
        try:
            save_array = np.where(save_array>1, 1, 0)
            ## save the result
            print(save_array.shape, np.unique(save_array))
            new_name = names.split("_0000")[0] + ".nii.gz"
            save_img = sitk.GetImageFromArray(save_array)
            save_img = sitk.Cast(save_img, sitk.sitkInt32)
            # os.rename(os.path.join(save_path, names+".nii.gz"), os.path.join(save_path, new_name))
            sitk.WriteImage(save_img, os.path.join(save_path, f"{new_name}.nii.gz"))
        except:
            print(f"{names} has error")
    else:
        # save_img = sitk.ReadImage(os.path.join(path, names, nii_files))
        # # Cast the image to int
        # save_img = sitk.Cast(img, sitk.sitkInt32)
        # sitk.WriteImage(save_img, os.path.join(save_path, f"{new_name}.nii.gz"))
        print(f"{names} skips")
            