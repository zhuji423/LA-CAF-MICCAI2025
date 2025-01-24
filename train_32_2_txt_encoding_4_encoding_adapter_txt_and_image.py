import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import random
import warnings
warnings.filterwarnings("ignore")
import logging
import sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from model.Universal_model_4adapter_txt_image import Universal_model_adapter
from dataset.dataloader import get_loader
from utils import loss_2_txt_encoding as loss
from utils.utils_4_txt_encoding import dice_score, check_data, TEMPLATE_vein, get_key, NUM_CLASS,ORGAN_NAME
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
torch.multiprocessing.set_sharing_strategy('file_system')
from monai.utils import set_determinism
from validation_4_txt_encoding import validation
# set_determinism(seed=1307)

def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name']
        logit_map = model(x)
        term_seg_Dice = loss_seg_DICE.forward(logit_map, y[:,:,:,:,:], name, TEMPLATE_vein)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y[:,:,:,:,:], name, TEMPLATE_vein)
        loss = term_seg_BCE + term_seg_Dice
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        # logging.info("Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item()))
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    logging.info(
                'Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)
   

def process(args,snapshot_path):
    world_size = 1
    rank = args.local_rank
    print(f"i am running process on rank {rank}")
    if args.dist:
        # args.local_rank = 0
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        dist.init_process_group(backend="nccl", init_method="env://")
        
    args.device = torch.device(f"cuda:{rank}")
    # args.device = "cuda:%d" % rank
    
    torch.cuda.set_device(args.device)

    # prepare the 3D model
    model = Universal_model_adapter(img_size=(args.roi_x, args.roi_y, args.roi_z),
    # model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),

                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding=args.trans_encoding
                    )

    #Load pre-trained weights
    if args.pretrain is not None:
        store_dict = model.state_dict()
        checkpoint = torch.load(args.pretrain)
        load_dict = checkpoint['net']
        num_count = 0
        for key, value in load_dict.items():
            if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
                name = '.'.join(key.split('.')[1:])
                name = 'backbone.' + name
            else:
                name = '.'.join(key.split('.')[1:])
            store_dict[name] = value
            num_count += 1
        ## 新加的两个类别 使用线性投射层
        # transposed = store_dict['organ_embedding'].transpose(0, 1)
        # linear = nn.Linear(32, 2).to(args.device)
        # organ_embedding = linear(transposed.float())  # shape is now [512, 2]
        # store_dict['organ_embedding'] = organ_embedding.transpose(0, 1)  # shape is now [2, 512]

        zeros_tensor = torch.zeros([4, 512]).to(args.device)
        store_dict['organ_embedding'] = zeros_tensor
        model.load_state_dict(store_dict)
        print('Use pretrained weights. load', num_count, 'params into', len(store_dict.keys()))


    if args.trans_encoding == 'word_embedding':

        ## using pretrained weight to transform word_embedding
        # word_embedding = torch.load(args.word_embedding)
        # transposed = word_embedding.transpose(0, 1)
        # linear = nn.Linear(32, 2).to(args.device)
        # organ_embedding = linear(transposed.float())  # shape is now [512, 2]
        # model.organ_embedding.data = organ_embedding.transpose(0, 1)  # shape is now [2, 512]

        ### using new clip word embedding    
        word_embedding = torch.load(args.word_embedding)
        model.organ_embedding.data = word_embedding.float()
        print('load word embedding')
        # print("cause pretrained model has been loaded, so no need to load word embedding")

    model.to(args.device)
    model.train()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device],find_unused_parameters=True)

    # criterion and optimizer
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    if args.backbone == 'unetpp':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                              nesterov=False, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler = get_loader(args)
    

    if rank == 0:
        writer = SummaryWriter(log_dir=snapshot_path)
        print('Writing Tensorboard logs to ', snapshot_path)

    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE) ## 单纯train一个epoch
        if rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)
        ##################validation and test start ################
        if (args.epoch% args.store_num == 0 and args.epoch != 0) and rank == 0:
        # if (args.epoch > args.max_epoch*0.1 and  args.epoch% args.store_num == 0 and args.epoch != 0) and rank == 0:

            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            # torch.save(checkpoint, snapshot_path+ \
            #             '/epoch_' + str(args.epoch) +'.pth')
            # print('save model success')
            args.phase = 'validation'
            validation_loader, val_transforms = get_loader(args)
            val_model_mean_dice = validation(model, validation_loader, args, args.epoch,snapshot_path)

            args.phase = 'test'
            test_loader, _ = get_loader(args)
            test_model_mean_dice = validation(model, test_loader, args, args.epoch,snapshot_path)

            # if not os.path.isdir('/pub/data/yangdeq/CLIP/data/vein/' + args.log_name +"/"+ args.copy):
            #     os.mkdir('/pub/data/yangdeq/CLIP/data/vein/' + args.log_name +"/"+ args.copy)
            torch.save(checkpoint, snapshot_path+ \
                        '/epoch_' + str(args.epoch) + '_validation_DICE_AVG:' + str(np.round(val_model_mean_dice,4)) + 
                            '_test_DICE_AVG:' + str(np.round(test_model_mean_dice,4)) +
                            '.pth')
            print('save model success')
            args.phase = 'train'
        ##################validation and test  end ################
        args.epoch += 1

    dist.destroy_process_group()

def main():
    # torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '24332'
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local-rank", type=int,default=0)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=1)
    ## logging
    parser.add_argument('--log_name', default='model2024/0722/vein_semi_full_717_4classes_attention_512_concat_4txt_img_adapter', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default=None,\
                         help='The path resume from checkpoint')##r"/pub/data/yangdeq/CLIP/data/vein/out/vein_swinunetr_new_clip_400/epoch_360.pth"
    parser.add_argument('--pretrain', default="/pub/data/yangdeq/CLIP/pretrained_weight/unet.pth",  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                        help='The path of pretrain model. ')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='/pub/data/yangdeq/CLIP/pretrained_weight/txt_encoding_4_artery_vein.pth', 
                        help='The path of word embedding')
    # /home/yangdq/project/module/CLIP-Driven-Universal-Model/pretrained_weights/txt_encoding.pth
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1001, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=8e-4, type=float, help='Learning rate') ## 1e-4 4e-10 used for continue training
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset  all the preprocess ok
    parser.add_argument('--dataset_list', nargs='+', default=["vein_semi"]) # 'PAOT', 'felix'
    parser.add_argument('--train_txt', nargs='+', default='_full_train.txt')#_train_hessian_full.txt
    parser.add_argument('--val_txt', nargs='+', default='_full_val.txt') #_val_hessian_full.txt
    parser.add_argument('--test_txt', nargs='+', default='_full_test.txt') #_test_hessian_full.txt
    parser.add_argument('--data_root_path', default='/pub/data/yangdeq/CLIP/data/vein/', help='data root path')
    parser.add_argument('--data_txt_path', default='/home/yangdq/project/module/CLIP-Driven-Universal-Model/dataset/vein_semi_full_list/', help='data txt path')
    ##############################################
    parser.add_argument('--batch_size', default=4, help='batch size')
    parser.add_argument('--num_workers', default=16, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-700, type=float, help='a_min in ScaleIntensityRanged') ##-175 -700
    parser.add_argument('--a_max', default=300, type=float, help='a_max in ScaleIntensityRanged') ##250 300
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    # parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
                                            '07', '08', '09', '12', '13', '10_03', 
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=True, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0, type=float, help='The percentage of cached data in total')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')
    parser.add_argument('--copy', default='1', type=str, help='current experiment description')
    args = parser.parse_args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    print(args) 
    snapshot_path = f'/pub/data/yangdeq/CLIP/data/vein/{args.log_name}/{args.backbone}_epoch:{args.max_epoch}_batch_size:{args.batch_size}_{args.lr}/{args.copy}'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    process(args=args,snapshot_path=snapshot_path)

if __name__ == "__main__":
    print("start training")
    main()

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py --dist True --uniform_sample
# CUDA_VISIBLE_DEVICES=0,1 python -W ignore -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 train.py --dist True --data_root_path /mnt/zzhou82/PublicAbdominalData/ --num_workers 12 --num_samples 4 --cache_dataset --cache_rate 0.6 --uniform_sample