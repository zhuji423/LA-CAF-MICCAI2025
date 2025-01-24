import os
import SimpleITK as sitk
import numpy as np
from medpy import metric

def dice_coefficient(prediction, target, class_num = 16):

    dice_coefficient = []
    a = np.unique(target)

    for i in range(class_num - 1):
            dice_cls = metric.binary.dc(prediction == (i + 1), target == (i + 1))
            dice_coefficient.append(dice_cls)
    print("ojbk")

    return sum(dice_coefficient)/len(dice_coefficient)

def jaccard_coefficient(prediction, target,class_num = 16 ):

    jaccard_coefficient = []
    a = np.unique(target)

    for i in range(class_num - 1):
        dice_cls = metric.binary.jc(prediction == (i + 1), target == (i + 1))
        jaccard_coefficient.append(dice_cls)

    return sum(jaccard_coefficient)/len(jaccard_coefficient)

def hurface_distance(prediction, target, slice,class_num = 14):

    hurface_distance = []
    a = np.unique(target)

    if  slice == 9 or slice == 5:
        print('第九张到了')
        for i in range(class_num - 1):
            print(i)
            if i == 6 or i == 7:
                continue
            else:
                dice_cls = metric.binary.hd95(prediction == (i + 1), target == (i + 1))
                hurface_distance.append(dice_cls)

    elif slice == 14:
        print('第14张到了')
        for i in range(class_num - 1):
            if i == 8 or i == 9:
                continue
            else:
                dice_cls = metric.binary.hd95(prediction == (i + 1), target == (i + 1))
                hurface_distance.append(dice_cls)

    elif slice == 7:
        print('第7张到了')
        for i in range(class_num - 1):
            if i == 7:
                continue
            else:
                dice_cls = metric.binary.hd95(prediction == (i + 1), target == (i + 1))
                hurface_distance.append(dice_cls)

    elif slice == 10:
        print('第10张到了')
        for i in range(class_num - 1):
            if i == 6:
                continue
            else:
                dice_cls = metric.binary.hd95(prediction == (i + 1), target == (i + 1))
                hurface_distance.append(dice_cls)


    elif slice == 11 or slice == 12 or slice == 13:
        for i in range(class_num - 1):
            if i == 8:
                continue
            else:
                dice_cls = metric.binary.hd95(prediction == (i + 1), target == (i + 1))
                hurface_distance.append(dice_cls)

    else:
        for i in range(class_num - 1):
            dice_cls = metric.binary.hd95(prediction == (i+1), target == (i+1))
            hurface_distance.append(dice_cls)

    return sum(hurface_distance)/len(hurface_distance)

def a_distance(prediction, target, slice, class_num = 14):

    a_distance = []
    a = np.unique(target)

    if slice == 14:
        print('第14张到了')
        for i in range(class_num - 1):
            if i == 8 or i == 9:
                continue
            else:
                print(i)
                dice_cls = metric.binary.asd(prediction == (i + 1), target == (i + 1))
                a_distance.append(dice_cls)

    elif slice == 9 or slice == 5:
        print('第九张到了')
        for i in range(class_num - 1):
            print(i)
            if i == 6 or i == 7:
                continue
            else:
                dice_cls = metric.binary.asd(prediction == (i + 1), target == (i + 1))
                a_distance.append(dice_cls)

    elif slice == 7:
        print('第7张到了')
        for i in range(class_num - 1):
            if i == 7:
                continue
            else:
                dice_cls = metric.binary.asd(prediction == (i + 1), target == (i + 1))
                a_distance.append(dice_cls)

    elif slice == 10:
        print('第10张到了')
        for i in range(class_num - 1):
            if i == 6:
                continue
            else:
                dice_cls = metric.binary.asd(prediction == (i + 1), target == (i + 1))
                a_distance.append(dice_cls)



    elif slice == 11 or slice == 12 or slice == 13:
        for i in range(class_num - 1):
            if i == 8:
                continue
            else:
                dice_cls = metric.binary.asd(prediction == (i + 1), target == (i + 1))
                a_distance.append(dice_cls)

    else:
        for i in range(class_num - 1):
            dice_cls = metric.binary.asd(prediction == (i+1), target == (i+1))
            a_distance.append(dice_cls)

    return sum(a_distance)/len(a_distance)

def evaluate_segmentation(predictions, labels):
    dice_scores = []
    jaccard_scores = []
    asd_scores = []
    hd95_scores = []
    slice = 1

    for pred_path, label_path in zip(predictions, labels):
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))


        dice_scores.append(dice_coefficient(pred, label))
        jaccard_scores.append(jaccard_coefficient(pred, label, slice))
        asd_scores.append(a_distance(pred, label, slice))
        hd95_scores.append(hurface_distance(pred, label, slice))
        slice += 1

    avg_dice = np.mean(dice_scores)
    avg_jaccard = np.mean(jaccard_scores)
    avg_asd = np.mean(asd_scores)
    avg_hd95 = np.mean(hd95_scores)

    return avg_dice, avg_jaccard, avg_asd, avg_hd95
    # return avg_dice, #avg_jaccard

def calculate_metrics(pred_folder,label_folder):
    pred_files = [os.path.join(pred_folder, file) for file in os.listdir(pred_folder)]
    label_files = [os.path.join(label_folder, file) for file in os.listdir(label_folder)]


    avg_dice, avg_jaccard, avg_asd, avg_hd95 = evaluate_segmentation(pred_files, label_files)
    # avg_dice= evaluate_segmentation(pred_files, label_files)

    print("Average Dice:", avg_dice)
    print("Average Jaccard:", avg_jaccard)
    # return avg_dice#,avg_jaccard
    print("Average ASD:", avg_asd)
    print("Average HD95:", avg_hd95)

