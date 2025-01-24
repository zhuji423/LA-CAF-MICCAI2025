import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import argparse
from monai.inferers import sliding_window_inference
from model.Compounded_unet import UNET3D_com
from dataset.dataloader import get_loader_without_gt
from utils.utils_4_txt_encoding import  threshold_organ
from utils.utils_4_txt_encoding import  NUM_CLASS
from utils.utils_4_txt_encoding import threshold_organ, save_results_4_veins

torch.multiprocessing.set_sharing_strategy('file_system')



def validation(model, ValLoader, val_transforms, args,save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.eval()
    for index, batch in enumerate(tqdm(ValLoader)):
        image, name = batch["image"].cuda(), batch["name"] # torch.Size([1, 1, 279, 210, 294])

        with torch.no_grad():
            pred = sliding_window_inference(image, 
                                            (args.roi_x, args.roi_y, args.roi_z), 
                                            1,
                                            model, 
                                            overlap=0.5,
                                            mode='gaussian',
                                            )
            pred_sigmoid = F.sigmoid(pred)
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        # pred_hard_summed = torch.sum(pred_hard, dim=1).unsqueeze(1)
        # pred_hard_post_pseudo_label_all = torch.where(pred_hard> 1, torch.tensor(1), torch.tensor(0))
        pred_save = torch.where(pred_hard>0.5,1,0)
        torch.cuda.empty_cache()
        for i in range(1,pred_save.shape[1]+1):
            pred_save[0, i-1, pred_save[0, i-1] > 0] = i
        # organ_list = [i for i in range(1,5)]  ###using for vessel
        print(f"pred_hard_post_pseudo_label max {pred_hard.shape}")
        pred_save_merge = pred_save.sum(axis = 1).unsqueeze(1)
        pred_save_merge[pred_save_merge>4] = 0
        batch['results'] = pred_save_merge
        print(batch.keys())
        save_results_4_veins(batch, save_path, val_transforms, [1])
        print("save path")
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=0)
    parser.add_argument('--resume', default='/root/project_lvm/weights/', help='The path resume from checkpoint')
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet]')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    parser.add_argument('--a_min', default=-700, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=300, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type= float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    ## dataset
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')
    parser.add_argument('--data_root_path', default="/root/project_lvm/ct/", help='data root path')
    parser.add_argument('--result_save_path', default="/root/project_lvm/RESULT/", help='path for save result')
    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--threshold', default=0.6, type=float)

    args = parser.parse_args()

    # prepare the 3D model
    model = UNET3D_com(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding='word_embedding'
                    )
    for pth in os.listdir(args.resume):
        if 'epoch' in pth:
            epoch = pth.split('_test')[0]
            resume_path = os.path.join(args.resume, pth)
            save_path = os.path.join(args.result_save_path, epoch)
            # if not os.path.exists(save_path):  
                #Load pre-trained weights
            store_dict = model.state_dict()
            checkpoint = torch.load(resume_path)
            load_dict = checkpoint['net']
            num_count = 0
            model.load_state_dict(load_dict)
            print('Use pretrained weights. load', num_count, 'params into', len(store_dict.keys()),"name",args.resume)
            model.cuda()
            torch.backends.cudnn.benchmark = True
            test_loader, val_transforms = get_loader_without_gt(args)
            validation(model, test_loader, val_transforms, args,save_path)
            # else:
            #     print("results already exist")
if __name__ == "__main__":
    main()
