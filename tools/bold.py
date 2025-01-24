import numpy as np
import nibabel as nib
from scipy import ndimage
import os

# 加载 NIfTI 文件
def label_bold(file_path, output_file):
    nii_file = file_path
    img = nib.load(nii_file)
    data = img.get_fdata()
    unique_labels = np.unique(data)[1:]
    print(f"unique_labels: {unique_labels}")
    # 定义要加粗的标签值列表
    # label_values = [1, 2, 3, 4]  # 用您要加粗的标签值列表替换
    
    # 对每个标签值执行边缘检测和像素添加
    for label_value in unique_labels:
        # 创建标签的二进制掩码
        mask = (data == label_value)
    
        # 使用 Sobel 滤波器找到边缘
        edges = np.zeros_like(data)
        for axis in range(3):
            edges += np.abs(ndimage.sobel(mask.astype(float), axis=axis))
        # 将边缘二值化
        edges = (edges > 0)
        # 使用二进制膨胀在边缘周围添加像素点
        expanded_edges = ndimage.binary_dilation(edges, structure=np.ones((1, 1, 1)))
        # 将添加的像素点应用到标签数据
        data[expanded_edges] = label_value
        
    data = data.astype(np.int64)

    # 将修改后的数据保存到新的 NIfTI 文件
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, output_file)
    print(f"save to {output_file}")
    return 0
# path = r"/pub/data/yangdeq/air_seg/vein/nnUNet_raw/Dataset004_vein_semi_seg_640/labelsTr/"
# save_path = r"/pub/data/yangdeq/air_seg/vein/nnUNet_raw/Dataset005_vein_semi_seg_640_mask/labelsTr/"
path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/semi_full_labelsTr_ori/"
save_path = r"/pub/data/yangdeq/CLIP/data/vein/25_vein_artery/semi_full_labelsTr_bold/"
os.makedirs(save_path, exist_ok=True)
for label_img in os.listdir(path):
    # if label_img in selected_label_list:
        label_path = path + label_img
        label_bold(label_path,save_path + label_img)