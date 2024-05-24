import os
import numpy as np
import SimpleITK as sitk
from MetricsReloaded.metrics.pairwise_measures import MultiClassPairwiseMeasures
import scipy
from collections import Counter
from nnunet.evaluation.surface_dice import normalized_surface_dice
pred_path ='/home/wangyitao/nnunet_train/visualization/nnmamba'
gt_path = '/mntnfs/med_data5/AMOS22/AMOS/DATASET/AMOS22_raw/nnUNet_raw_data/Task001_AMOS_CT/labelsTr'
pred_list = os.listdir(pred_path)
class_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
total_sum = []
print(pred_list)
def compute_surface_dice_at_tolerance(surface_distances, tolerance_mm):
  distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
  distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
  surfel_areas_gt      = surface_distances["surfel_areas_gt"]
  surfel_areas_pred    = surface_distances["surfel_areas_pred"]
  overlap_gt   = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
  overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
  surface_dice = (overlap_gt + overlap_pred) / (
      np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
  return surface_dice

for i in pred_list:
    pred_nii  = sitk.ReadImage(os.path.join(pred_path,i))
    pred_img = sitk.GetArrayFromImage(pred_nii)
    gt_nii  = sitk.ReadImage(os.path.join(gt_path,i))
    gt_img = sitk.GetArrayFromImage(gt_nii)
    print(i)
    spacing = pred_nii.GetSpacing()
    sum_single = 0
    single_avr = normalized_surface_dice(pred_img,gt_img,0.5)
    print('single_prediction_nsd:',single_avr)
    total_sum.append(single_avr)
print('mNSD:',np.average(total_sum))
        