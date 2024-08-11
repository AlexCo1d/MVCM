"""
VQA task training script
"""

import argparse
import os
import sys

import transformers

from Utils.pos_embed import interpolate_pos_embed

import time
import datetime
import json
from pathlib import Path
import torch
import torch.distributed as dist
import utils
import Utils.misc as misc
import Utils.lr_sched as lr_sched
from Dataset import create_dataset
from model.MVCM_VQA import MVCM_VQA
from vqaTools.vqaEvaluate import compute_vqa_acc


def train(model, data_loader, optimizer, epoch, device, args):
    # train
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10
    for i, b in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args)
        if args.distill_model:
            alpha = 0.4
            alpha = alpha * min(1, (epoch * len(data_loader) + i) / len(data_loader))
            loss = model(b, dataloader=data_loader, alpha=alpha)
        else:
            loss = model(b, dataloader=data_loader)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.mean().item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, device, args):
    """
    :param model: VQA model
    :param data_loader: test_loader
    :param device: device
    :param args: arguments
    :return: a dict that contains the result of the evaluation {question, pred, answer, answer_type}
    """
    # test
    model.eval()
    if args.distributed:
        t_model = model.module
        t_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate VQA test result:'
    print_freq = 10

    result = []

    for n, b in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        text_output = t_model.predict_answers(b, dataloader=data_loader)
        for idx, answer in enumerate(text_output):
            # 构造结果字典
            result_dict = {
                'image_name': b['image_name'][idx],  #获取图片名
                "question": b['text_input'][idx],  # 当前问题
                "pred": text_output[idx],  # 预测的答案
                "answer": b['text_output'][idx],  # 实际答案
                "answer_type": b['answer_type'][idx]  # 答案类型
            }
            result.append(result_dict)
    model.train()
    return result


def main(args):
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    utils.set_seed(args.seed + utils.get_rank())

    #### Loading Dataset ####
    print('Creating vqa {} datasets'.format(args.dataset_use))
    train_dataset, test_dataset, _ = create_dataset(args)
    print('train dataset size: ', len(train_dataset))
    print('test dataset size: ', len(test_dataset))
    print('answer_list size: ', len(test_dataset.answer_list))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_test = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        # print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler_train, batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=sampler_test, batch_size=args.eval_batch_size,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    #### Creating Model ####
    print("Creating model")
    if args.classifier_vqa:
        model = MVCM_VQA(img_size=args.img_size, vit_type=args.vit_type, freeze_vit=args.freeze_vit,
                         vit_path=args.vit_path if args.checkpoint is None else '', distill=args.distill_model, bert=args.bert_type)

    misc.model_structure(model)
    model = model.to(device)
    # print(model)

    # eff_batch_size = args.batch_size * misc.get_world_size()

    # set group:

    proj = list(map(id, model.text_decoder.parameters()))
    proj_params = filter(lambda x: id(x) in proj, model.parameters())
    rest_params = filter(lambda x: id(x) not in proj, model.parameters())
    params = [
        {'params': proj_params, 'lr': args.lr * 3},
        {'params': rest_params}
    ]

    optimizer = torch.optim.AdamW(params=params, lr=args.lr, weight_decay=0.05) \
        if not args.deepspeed else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

        if args.deepspeed:
            import deepspeed
            model, optimizer, _, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer)

        model_without_ddp = model.module

    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # model_dict = model_without_ddp.state_dict()
        interpolate_pos_embed(model_without_ddp.visual_encoder, checkpoint, 'visual_encoder.pos_embed')
        # pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict and v.shape==model_dict[k].shape}
        # model_dict.update(pretrained_dict)
        # msg = model_without_ddp.load_state_dict(model_dict)
        msg = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
        if 'optimizer' in checkpoint and args.load_optim:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if args.start_epoch != -1:
            start_epoch = args.start_epoch

    start_time = time.time()
    if args.evaluate:
        print("\nStart evaluation\n")

        vqa_result = evaluation(model, test_loader, device, args)
        json.dump(vqa_result,
                  open(os.path.join(args.result_dir, 'vqa_result_%s.json' % (args.dataset_use)), 'w'))
        acc = compute_vqa_acc(vqa_result, args=args, dataloader=test_loader, epoch=checkpoint['epoch'])
        print(f'{args.dataset_use} acc: {acc}')
    else:
        print("\nStart training\n")
        for epoch in range(start_epoch, args.epochs):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            utils.cosine_lr_schedule(optimizer, epoch, args.epochs, args.lr, args.min_lr)

            train(model, train_loader, optimizer, epoch, device, args)

            if utils.is_main_process() and args.output_dir:

                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                prefix = args.checkpoint.split('/')[-1].split('.')[0]
                # for evaluation and output the result
                if epoch >= args.warmup_epochs and (epoch % args.eval_freq == 0 or epoch >= args.epochs - 1 or epoch % 5 == 0):
                    torch.save(save_obj,
                               os.path.join(args.output_dir, '%s_%s_%02d.pth' % (prefix, args.dataset_use, epoch)))

                    if args.dataset_use != 'pmcvqa':
                        vqa_result = evaluation(model, test_loader, device, args)
                        json.dump(vqa_result,
                                  open(os.path.join(args.result_dir, '%s_vqa_result_%s.json' % (prefix, epoch)), 'w'))
                        acc = compute_vqa_acc(vqa_result, args=args, dataloader=test_loader, epoch=epoch)
                        print({'acc:': acc})
                        json.dump({'acc:': acc},
                                  open(os.path.join(args.result_dir, 'vqa_metric.json'), 'a'))
                else:
                    torch.save(save_obj, os.path.join(args.output_dir, 'last_epoch_weight.pth'))
                del save_obj

            if args.distributed:
                dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_use', default='radvqa',
                        help='choose medical vqa dataset(radvqa, pathvqa, slake, vqa2019)')
    parser.add_argument('--dataset_path', help='path to the dataset')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--load_optim', action='store_true')
    parser.set_defaults(load_optim=False)

    parser.add_argument('--LLM_path', default='', type=str, help='path for loading pretrained LLM model')
    parser.add_argument('--is_lora', action='store_true')
    parser.set_defaults(is_lora=False)
    parser.add_argument('--classifier_vqa', action='store_true')
    parser.set_defaults(classifier_vqa=False)
    parser.add_argument('--distill_model', action='store_true')
    parser.set_defaults(distill_model=False)

    parser.add_argument('--vit_path', default='',
                        help='path for loading pretrained ViT model')
    parser.add_argument('--vit_type', default='eva_vit', type=str)
    parser.add_argument('--bert_type', type=str, default='bert-base-uncased',
                        help='type of bert model to use')
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--freeze_vit', action='store_true')
    parser.set_defaults(freeze_vit=False)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_batch_size', default=5, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.set_defaults(evaluate=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=20, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--output_dir', default='./output_dir', type=str)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--start_epoch', default=-1, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # deepspeed
    parser.add_argument('--deepspeed', action='store_true', help='use DeepSpeed for distributed training')
    parser.set_defaults(deepspeed=False)
    parser.add_argument('--deepspeed_config', type=str, default='./ds_config.json', help='DeepSpeed configuration file')

    args = parser.parse_args()

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    # set log, set console print info to file
    # sys.stdout = utils.Logger(filename=os.path.join(args.output_dir, "log.txt"), stream=sys.stdout)

    print("args: ", args)
    main(args)
