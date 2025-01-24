import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import glob
import json
from monai.inferers import sliding_window_inference

from model.Universal_model_4adapter_txt_image import Universal_model_adapter
# from model.Universal_model import Universal_model,Universal_model_2_txt_encoder
from dataset.dataloader import get_loader
from utils import loss
from utils.utils_4_txt_encoding import dice_score,jaccard_index, TEMPLATE,TEMPLATE_vein, ORGAN_NAME_VEIN,save_results_4_veins , get_key, NUM_CLASS
from utils.utils import extract_topk_largest_candidates, organ_post_process, threshold_organ,threshold_organ_4_txt_encoder
from collections import defaultdict
torch.multiprocessing.set_sharing_strategy('file_system')
from monai.metrics import *
from medpy import metric
from utils.compute_NSD import compute_NSD_metric
import seg_metrics.seg_metrics as sg
def validation(model, ValLoader, args, idx,snapshot_path,val_transforms):
    model.eval()
    dice_list = {}
    nsd_list = {}
    jaccard_list = {}
    hd95_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
        nsd_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
        jaccard_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
        hd95_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    save_dice_log = defaultdict(list)
    for index, batch in enumerate(tqdm(ValLoader)):
        # print('%d processd' % (index))
        image, label, name, img_name = batch["image"].cuda(), batch["post_label"], batch["name"], batch['img_name'],
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)
        # pred_hard = threshold_organ(pred_sigmoid)
        if pred_sigmoid.shape[1] == 4:
            pred_hard = threshold_organ_4_txt_encoder(pred_sigmoid)
        elif pred_sigmoid.shape[1] == 34:
            pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()
        B = pred_sigmoid.shape[0]
        for b in range(B):
            content = 'case %s| label %s' %(img_name[b], name[b] )
            template_key = get_key(name[b])
            organ_list = TEMPLATE_vein[template_key]
            if pred_sigmoid.shape[1] == 4 :
                organ_list = TEMPLATE_vein[template_key]
                pred_hard_post = pred_sigmoid
            elif pred_sigmoid.shape[1] == 32:
                pred_hard = threshold_organ(pred_sigmoid)
                pred_hard_post = organ_post_process(pred_hard.cpu().numpy(), organ_list, args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1],args)
                pred_hard_post = torch.tensor(pred_hard_post)

            # for organ in organ_list:
            #     if torch.sum(label[b,32 + organ-1,:,:,:]) != 0:
            #         dice_organ, recall, precision = dice_score(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,32+organ-1,:,:,:].cuda())
            #         jaccard = jaccard_index(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,32+organ-1,:,:,:].cuda())
            #         nsd,hd95 = compute_NSD_metric(pred_hard_post[b,organ-1,:,:,:].cpu().numpy(), label[b,32+organ-1,:,:,:].cpu().numpy())
            #         dice_list[template_key][0][organ-1] += dice_organ.item()
            #         dice_list[template_key][1][ organ-1] += 1

            #         nsd_list[template_key][0][organ-1] += nsd.item()
            #         nsd_list[template_key][1][ organ-1] += 1

            #         jaccard_list[template_key][0][organ-1] += jaccard.item()
            #         jaccard_list[template_key][1][ organ-1] += 1

            #         hd95_list[template_key][0][organ-1] += hd95.item()
            #         hd95_list[template_key][1][ organ-1] += 1

            #         content += ' %s: %.4f, '%(ORGAN_NAME_VEIN[ organ-1], dice_organ.item())
            #         save_dice_log[content] = ' %s: dice %.4f, jaccard %.4f, NSD %.4f, HD95 %.4f'%(ORGAN_NAME_VEIN[ organ-1], dice_organ.item(), jaccard,nsd,hd95)
            #         print(' %s: dice %.4f, jaccard %.4f, NSD %.4f, HD95 %.4f'%(ORGAN_NAME_VEIN[ organ-1], dice_organ.item(), jaccard,nsd,hd95))
            pred_save = torch.where(pred_hard_post>0.5,1,0)
            label_map = [1,2,1,2]
            for i in range(1,pred_save.shape[1]+1):
                pred_save[0, i-1, pred_save[0, i-1] > 0] = label_map[i-1]

            print(f"pred_hard_post_pseudo_label max {pred_hard.shape}")
            pred_save_merge = pred_save.sum(axis = 1).unsqueeze(1)
            pred_save_merge[pred_save_merge>2] = 0
            batch['results'] = pred_save_merge
            print(batch.keys())
            print("save path")
            save_path = os.path.join(snapshot_path,str(idx))
            save_results_4_veins(batch, save_path, val_transforms, [1])
        torch.cuda.empty_cache()

    ave_organ_dice = np.zeros((2, NUM_CLASS))
    ave_organ_jacc = np.zeros((2, NUM_CLASS))
    ave_organ_nsd = np.zeros((2, NUM_CLASS))
    ave_organ_hd95 = np.zeros((2, NUM_CLASS))


    with open(snapshot_path + f'/test_metrics_{idx}.json', 'a+') as f:
        for key in TEMPLATE_vein.keys():
            if key == "25":
                organ_list = TEMPLATE_vein[key]
                content = 'Task%s| '%(key)
                # content1 = 'NSD Task%s| '%(key)
                for organ in organ_list:
                    dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1] ##当前这个器官的平均dice，除以这个器官出现的次数
                    content += '%s: %.4f, '%(ORGAN_NAME_VEIN[organ-1], dice)
                    ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                    ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]

                    ave_organ_jacc[0][organ-1] += jaccard_list[key][0][organ-1]
                    ave_organ_jacc[1][organ-1] += jaccard_list[key][1][organ-1]

                    ave_organ_nsd[0][organ-1] += nsd_list[key][0][organ-1]
                    ave_organ_nsd[1][organ-1] += nsd_list[key][1][organ-1]

                    ave_organ_hd95[0][organ-1] += hd95_list[key][0][organ-1]
                    ave_organ_hd95[1][organ-1] += hd95_list[key][1][organ-1]

                print(content)
                f.write('\n')
                f.write("##################################we are validating " + str(idx) + " epoch##################################") 
                f.write('\n')
                f.write("we are in the " + args.phase + " phase") 
                f.write('\n') 
                f.write(content)
                f.write('\n')
        content = 'Average | '
        # for i in range(NUM_CLASS): ### original
        for i in range(4): ### only write artery and vein
            content += '%s: %.4f, '%(ORGAN_NAME_VEIN[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
        print(content)
        f.write(content)
        f.write('\n')

        print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
        f.write('%s: %.4f, \n'%('average artery dice', np.mean([i for i in ave_organ_dice[0] / ave_organ_dice[1] if not np.isnan(i) and i != 0][0::2])))
        f.write('%s: %.4f, \n'%('average vein dice', np.mean([i for i in ave_organ_dice[0] / ave_organ_dice[1] if not np.isnan(i) and i != 0][1::2])))
        
        f.write('%s: %.4f, \n'%('average  artery jaccard', np.mean([i for i in ave_organ_jacc[0] / ave_organ_jacc[1] if not np.isnan(i) and i != 0][0::2])))
        f.write('%s: %.4f, \n'%('average  vein jaccard', np.mean([i for i in ave_organ_jacc[0] / ave_organ_jacc[1] if not np.isnan(i) and i != 0][1::2])))
        
        f.write('%s: %.4f, \n'%('average artery nsd', np.mean([i for i in ave_organ_nsd[0] / ave_organ_nsd[1] if not np.isnan(i) and i != 0][0::2])))
        f.write('%s: %.4f, \n'%('average artery nsd', np.mean([i for i in ave_organ_nsd[0] / ave_organ_nsd[1] if not np.isnan(i) and i != 0][1::2])))
        
        f.write('%s: %.4f, \n'%('average artery hd95', np.mean([i for i in ave_organ_hd95[0] / ave_organ_hd95[1] if not np.isnan(i) and i != 0][0::2])))
        f.write('%s: %.4f, \n'%('average artery hd95', np.mean([i for i in ave_organ_hd95[0] / ave_organ_hd95[1] if not np.isnan(i) and i != 0][1::2])))
        
        f.write('\n')
        f.write(json.dumps(save_dice_log))
    model_dice = np.mean([i for i in ave_organ_dice[0] / ave_organ_dice[1] if not np.isnan(i)])
    return model_dice




def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='/pub/data/yangdeq/CLIP/data/vein/model2024/draw_for_pics/univer_clip', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--start_epoch', default=490, type=int, help='Number of start epoches')
    parser.add_argument('--end_epoch', default=490, type=int, help='Number of end epoches')
    parser.add_argument('--epoch_interval', default=100, type=int, help='Number of start epoches')
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet]')

    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['vein_artery']) # 'vein_artery' 'vein_semi_full_'
    parser.add_argument('--data_root_path', default='/pub/data/yangdeq/CLIP/data/vein/', help='data root path')
    parser.add_argument('--data_txt_path', default='/home/yangdq/project/module/CLIP-Driven-Universal-Model/dataset/vein_list/', help='data txt path') #vein_semi_full_list
    parser.add_argument('--train_txt', nargs='+', default="_train_hessian_full.txt")#_train_hessian_full.txt   'train.txt'
    parser.add_argument('--val_txt', nargs='+', default='_val_hessian_full.txt')  #_val_hessian_full.txt  val.txt
    parser.add_argument('--test_txt', nargs='+', default='_test_hessian_full.txt') #_test_hessian_full.txt  test.txt

    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-700, type=float, help='a_min in ScaleIntensityRanged') #-175 -700
    parser.add_argument('--a_max', default=300, type=float, help='a_max in ScaleIntensityRanged') #250   300
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')

    args = parser.parse_args()

    # prepare the 3D model
    model = Universal_model_adapter(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding='word_embedding'
                    )
    # model = Universal_model_2_txt_encoder(img_size=(args.roi_x, args.roi_y, args.roi_z),
    #                 in_channels=1,
    #                 out_channels=2,
    #                 backbone=args.backbone,
    #                 encoding='word_embedding'
    #                 )
    #Load pre-trained weights
    # store_path_root = f'/pub/data/yangdeq/CLIP/data/vein/' + args.log_name + '/epoch_*.pth'
    # store_path_root = f'/pub/data/yangdeq/CLIP/data/vein/model2024/vein_full_train_CLIP_univer_34_class/epoch_284_test_DICE_AVG:0.6031.pth'
    
    # store_path_root = f"/pub/data/yangdeq/CLIP/comperation_results/ours/epoch_500_validation_DICE_AVG:0.7629_test_DICE_AVG:0.76.pth"
    store_path_root = r"/pub/data/yangdeq/CLIP/data/vein/model2024/0715/vein_semi_full_717_4classes_attention_512_concat_4txt_img_adapter/unet_epoch:1001_batch_size:3_0.0008/1/epoch_10_validation_DICE_AVG:0.5871_test_DICE_AVG:0.5841.pth"
    print(store_path_root)
    for store_path in glob.glob(store_path_root):
        # if "400" not in store_path:
            print(f"i am  validation {store_path}")
            # if "swin" in store_path:
            # store_path = store_path_root
            store_dict = model.state_dict()
            load_dict = torch.load(store_path)['net']

            # for key, value in load_dict.items():
            #     if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
            #         name = '.'.join(key.split('.')[1:])
            #         # name = 'backbone.' + name
            #     else:
            #         name = '.'.join(key.split('.')[1:])
            #     store_dict[name] = value
            # organ_embedding = torch.rand(4, 512)
            # store_dict['organ_embedding'] = organ_embedding
            model.load_state_dict(load_dict)
            print(f'Load {store_path} weights')

            model.cuda()

            torch.backends.cudnn.benchmark = True
            validation_loader, val_transform = get_loader(args)
            # if "DICE" not in store_path:
            #     i = int(store_path.split('_')[-1].split('.')[0])#+1 ### no rename 
            #     model_mean_dice = validation(model, validation_loader, args, i)
            #     new_name = store_path.replace(str(i), str(i)+"_" + args.phase +'_DICE_AVG:'+str(np.round(model_mean_dice,4)))
            # else:
                # i = int(store_path.split('epoch_')[-1].split("_DICE")[0])
            # i =  int(store_path.split('epoch_')[-1][:3])
            i = 10
            # model_name = store_path.split('/')[-1].split('.')[0]
            snapshot_path = r"/pub/data/yangdeq/CLIP/comperation_results/univer_clip"
            model_mean_dice = validation(model, validation_loader, args, i, snapshot_path, val_transform)
            # new_name = store_path.split("epoch")[0] + 'epoch_' + str(i) + "_" + args.phase + '_DICE_AVG:' +  str(np.round(model_mean_dice,4)) + ".pth"
            
            
            # os.rename(store_path, new_name)

if __name__ == "__main__":
    main()

#python validation.py >> out/Nvidia/ablation_clip/clip2.txt