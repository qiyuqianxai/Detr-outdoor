import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from matplotlib import pyplot as plt
from PIL import Image
# 定义展示图片函数并展示几张图片
def show_image(image_path):
    plt.figure(figsize=(16, 10))
    plt.xticks([])
    plt.yticks([])
    image = Image.open(image_path)
    plt.imshow(image)

def get_args_parser(custom_param):
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=custom_param.lr, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=custom_param.batch_size, type=int)
    parser.add_argument('--weight_decay', default=custom_param.weight_decay, type=float)
    parser.add_argument('--epochs', default=custom_param.epochs, type=int)
    parser.add_argument('--one_epoch_steps', default=custom_param.one_epoch_steps, type=int)
    parser.add_argument('--lr_drop', default=custom_param.lr_drop, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=64, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=4, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--num_classes', default=custom_param.num_classes, type=int,
                        help='#classes in your dataset, which can override the value hard-coded in file models/detr.py')
    parser.add_argument('--dataset_file', default=custom_param.dataset_file)
    parser.add_argument('--coco_path', default=custom_param.coco_path, type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default=custom_param.output_dir,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default=custom_param.device,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=custom_param.seed, type=int)
    # parser.add_argument('--resume', default=custom_param.resume, help='resume from checkpoint')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=custom_param.num_workers, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=3, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

# 基本参数配置，主要配置这个就行
class config:
    lr = 1e-4 # 学习率
    lr_backbone = 1e-5 # backbone的学习率
    lr_drop = 50 # 学习率在多少个epoch衰减一次
    batch_size = 1
    weight_decay = 1e-4
    epochs = 100

    # dataset parameters
    num_classes = 5 # 类别数

    dataset_file = "custom" # 使用自定义的数据库函数，不需要更改

    coco_path = r"D:\dataset\MyDatasets" # 自定义数据集路径的路径

    device = 'cuda' if torch.cuda.is_available() else "cpu" # device to use for training / testing

    seed = 42 # 随机种子

    resume = "detr-r50_no-class-head.pth" # 'resume from checkpoint'

    num_workers = 2 # 读取数据的进程数

    output_dir = "outputs" # 保存模型的路径

def main(args):
    # 初始化分布式模型参数配置
    utils.init_distributed_mode(args)

    # 打印参数配置信息
    print(args)

    device = torch.device(args.device)

    # 固定随机种子以便于复现
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 读取模型并转到gpu或者cpu上
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # 将模型分为多卡分布式模型和单卡训练模型，根据需求使用模型
    model_without_ddp = model
    if args.distributed:
        # 使用分布式模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 获取并打印模型参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # 给backbone和主网络分别配置优化器
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # optimizer = torch.optim.SGD(param_dicts,lr=args.lr,momentum=0.9)

    # 设置学习率衰减策略
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 读取训练集，验证集
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # 根据是否采用分布式训练分配每一batch的数据
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        # 随机取样
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # 设置dataloader
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        # 对验证数据做单独处理，一般是不对验证集做数据增强
        base_ds = get_coco_api_from_dataset(dataset_val)

    # 如果冻结参数，则读取预训练权重进行训练
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # 保存路径
    output_dir = Path(args.output_dir)

    if args.resume:
        # 读取预训练模型配置，包括权重，学习率，epoch等参数，也可以读取之前自己未训练完成的模型继续训练
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # 如果是测试模式则直接读取模型测试就return
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    # 开始训练 #
    print("Start training")
    # 记录训练时间
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        # 训练一个epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        # 学习率衰减
        lr_scheduler.step()
        # 保存模型
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        # 测试模型在验证集上的效果
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        # 保存测试结果
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    # 显示训练总耗时
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

# 获取配置参数
# # parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
# # args = parser.parse_args(config())
# if __name__ == '__main__':
#     if args.output_dir:
#         Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#     main(args)
