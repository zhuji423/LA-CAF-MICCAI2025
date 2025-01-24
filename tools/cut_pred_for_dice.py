import SimpleITK as sitk
import numpy as np
import itertools
import cc3d
def count_consecutive_same_values(input_list):
    return [len(list(group)) for key, group in itertools.groupby(input_list)]

# Read the CT image and the right lung label
def compute_mask( infer_label,label):
    # infer_label_image = sitk.ReadImage(infer_label)
    # label_file_name = label
    # label_image = sitk.ReadImage(label_file_name)

    label_array = label
    infer_label_array = infer_label
    print(label_array.shape)
    sum_row = label_array.sum(axis=0).sum(axis=0)
    binary_array = sum_row > 0
    counter = count_consecutive_same_values(binary_array)
    try:
        assert len(counter) == 3
        print("counter:", counter)
    except:
        print(f"wrong counter:{counter}")
    
    mx_counter = max(counter)
    right_lung = 0
    # if np.sum(np.logical_not(np.array(counter) >= 256)) >= 1:
    if counter[0] == mx_counter: #right lung
        right_lung = 1
    elif counter[-1] == mx_counter: #left lung
        right_lung = 0

    Depth = label_array.shape[0]
    if right_lung:
        infer_label_array[:Depth//2, :, :] = 0
        cropped_label = infer_label_array
        print("right lung")

    else:
        infer_label_array[Depth//2:, :,:] = 0
        cropped_label = infer_label_array
        print("left lung")
        print(np.unique(cropped_label))
    # savemask = sitk.GetImageFromArray(cropped_label)
    # sitk.WriteImage(savemask,r"/home/yangdq/project/module/CLIP-Driven-Universal-Model/tools/artery_jj_pred_cut.nii.gz")
    return cropped_label