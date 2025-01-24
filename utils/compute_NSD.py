import numpy as np
import os
join = os.path.join
from collections import OrderedDict
from utils.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient,compute_robust_hausdorff
seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
label_tolerance = OrderedDict({'left artery': 2,'left vein': 2,'right artery': 2,'right artery': 2,})
for organ in label_tolerance.keys():
    seg_metrics['{}_DSC'.format(organ)] = list()
for organ in label_tolerance.keys():
    seg_metrics['{}_NSD'.format(organ)] = list()

def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.

    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int

    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) ==1, print('mask label error!')
    z_index = np.where(organ_mask>0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    
    return z_lower, z_upper


def compute_NSD_metric(preds, labels):
    preds = np.where(preds > 0.5, 1., 0.)
    gt_data = np.uint8(labels)
    seg_data = np.uint8(preds)

    organ_i_gt, organ_i_seg = gt_data, seg_data  
    DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)

    surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, (1.5,1.5,1.5))
    NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
    HD_95 = compute_robust_hausdorff(surface_distances, 95)
    return NSD_i,HD_95
