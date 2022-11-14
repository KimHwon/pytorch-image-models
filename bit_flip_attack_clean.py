#!/usr/bin/env python3

import argparse
import os
import sys
import csv
import glob
import json
import time
import logging
from unittest import result
import torch
import torch.nn as nn
import torch.nn.parallel
from tensorboardX import SummaryWriter
from collections import OrderedDict
from contextlib import suppress
import copy
from tqdm import tqdm

import pandas as pd
import numpy as np

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models, set_fast_norm
from timm.attack import BFA, hamming_distance
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser,\
    decay_batch_step, check_batch_size_retry

identifier_string = ''
def id_string(args):
    global identifier_string
    if identifier_string:
        return identifier_string
    
    identifier_string = '{}__{}__{}'.format(
        time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime(time.time())),
        f"{args.model}",
        f"b{args.batch_size}_n{args.n_iter}_k{args.k_top}"
    )
    return identifier_string

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('attack')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Bit-Flip Attack')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')
parser.add_argument('--num_gpu',
                    type=int,
                    default=1,
                    help='num_gpu')

# quantization
parser.add_argument(
    '--quan-bitwidth',
    dest='quan_bitwidth',
    type=int,
    default=8,
    help='the bitwidth used for quantization')
parser.add_argument(
    '--quan-reset-weight',
    dest='quan_reset',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
# Bit Flip Attack
parser.add_argument('--bfa',
                    dest='enable_bfa',
                    action='store_true',
                    help='enable the bit-flip attack')
parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')
parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')
parser.add_argument(
    '--k_top',
    type=int,
    default=10,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)
parser.add_argument('--random_bfa',
                    dest='random_bfa',
                    action='store_true',
                    help='perform the bit-flips randomly on weight bits')
parser.add_argument('--save_path',
                    type=str,
                    default='./output/',
                    help='Folder to save checkpoints and log.')

def main(args):
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    with suppress():
        # create model
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            in_chans=3,
            global_pool=args.gp,
            scriptable=args.torchscript,
            quan_bitwidth=args.quan_bitwidth,
            quan_reset=args.quan_reset)
        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes

        if args.checkpoint:
            load_checkpoint(model, args.checkpoint, args.use_ema)

        param_count = sum([m.numel() for m in model.parameters()])
        _logger.warning('Model %s created, param count: %d' % (args.model, param_count))

        data_config = resolve_data_config(
            vars(args),
            model=model,
            use_test_size=not args.use_train_size,
            verbose=True
        )
        model = model.cuda()

        if args.num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

        attacker = BFA(nn.CrossEntropyLoss().cuda(), model, args.k_top, gpu=1, logger=_logger)
        #model_clean = copy.deepcopy(model)

        train_dataset = create_dataset(
            root=os.path.join(args.data, 'train'), name=args.dataset, split=args.split,
            download=args.dataset_download, load_bytes=args.tf_preprocessing, class_map=args.class_map)
        val_dataset = create_dataset(
            root=os.path.join(args.data, 'val'), name=args.dataset, split=args.split,
            download=args.dataset_download, load_bytes=args.tf_preprocessing, class_map=args.class_map)

        train_loader = create_loader(
            train_dataset,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            tf_preprocessing=args.tf_preprocessing)
        val_loader = create_loader(
            val_dataset,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
            tf_preprocessing=args.tf_preprocessing)

        writer = SummaryWriter(os.path.join(args.save_path, f'run_{id_string(args)}', 'tb_log'))

        # Note that, attack has to be done in evaluation model due to batch-norm.
        # see: https://discuss.pytorch.org/t/what-does-model-eval-do-for-batchnorm-layer/7146
        model.eval()
        losses = AverageMeter()
        iter_time = AverageMeter()
        attack_time = AverageMeter()

        print(f"device_count : {torch.cuda.device_count()}")
        print(f"current_device : {torch.cuda.current_device()}")
        print(torch.cuda.memory_summary())

        # attempt to use the training data to conduct BFA
        for _, (data, target) in enumerate(train_loader):
            target = target.cuda(non_blocking=True)
            data = data.cuda()
            # Override the target to prevent label leaking
            _, target = model(data).data.max(1)
            break

        # evaluate the test accuracy of clean model
        val_acc_top1, val_acc_top5, val_loss, output_summary =  validate(val_loader, model, attacker.criterion,
                                                                        summary_output=True,
                                                                        gpu=0)
        
        tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
        tmp_df['BFA iteration'] = 0
        tmp_df.to_csv(os.path.join(args.save_path, f'run_{id_string(args)}', 'output_summary_BFA_0.csv'), index=False)

        writer.add_scalar('attack/val_top1_acc', val_acc_top1, 0)
        writer.add_scalar('attack/val_top5_acc', val_acc_top5, 0)
        writer.add_scalar('attack/val_loss', val_loss, 0)

        _logger.info('k_top is set to {}'.format(args.k_top))
        _logger.info('Attack sample size is {}'.format(data.size()[0]))
        end = time.time()

        df = pd.DataFrame()
        last_val_acc_top1 = val_acc_top1

        for i_iter in range(args.n_iter):
            _logger.warning('**********************************')
            if not args.random_bfa:
                attack_log = attacker.progressive_bit_search(model, data, target)
            else:
                attack_log = attacker.random_flip_one_bit(model)
            
            # measure data loading time
            attack_time.update(time.time() - end)
            end = time.time()

            # record the loss
            if hasattr(attacker, "loss_max"):
                losses.update(attacker.loss_max, data.size(0))
                
            _logger.warning(
                'Iteration: [{:03d}/{:03d}]   '
                'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
                format((i_iter + 1),
                    args.n_iter,
                    attack_time=attack_time,
                    iter_time=iter_time) + id_string(args))
            try:
                _logger.info('loss before attack: {:.4f}'.format(attacker.loss.item()))
                _logger.info('loss after attack: {:.4f}'.format(attacker.loss_max))
            except:
                pass
            
            _logger.info('bit flips: {:.0f}'.format(attacker.bit_counter))

            writer.add_scalar('attack/bit_flip', attacker.bit_counter, i_iter + 1)
            writer.add_scalar('attack/sample_loss', losses.avg, i_iter + 1)

            # exam the BFA on entire val dataset
            val_acc_top1, val_acc_top5, val_loss, output_summary = validate(val_loader, model, attacker.criterion,
                                                                            summary_output=True,
                                                                            gpu=0)
                                                                            
            tmp_df = pd.DataFrame(output_summary, columns=['top-1 output'])
            tmp_df['BFA iteration'] = i_iter + 1
            tmp_df.to_csv(os.path.join(args.save_path, f'run_{id_string(args)}', f'output_summary_BFA_{i_iter+1}.csv'), index=False)
        
            # add additional info for logging
            acc_drop = last_val_acc_top1 - val_acc_top1
            last_val_acc_top1 = val_acc_top1

            for i in range(attack_log.__len__()):
                attack_log[i].append(val_acc_top1)
                attack_log[i].append(acc_drop)

            df = df.append(attack_log, ignore_index=True)
            
            writer.add_scalar('attack/val_top1_acc', val_acc_top1, i_iter + 1)
            writer.add_scalar('attack/val_top5_acc', val_acc_top5, i_iter + 1)
            writer.add_scalar('attack/val_loss', val_loss, i_iter + 1)

            # measure elapsed time
            iter_time.update(time.time() - end)
            _logger.warning(
                'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                    iter_time=iter_time))
            end = time.time()

            if val_acc_top1 <= 0.2:
                break
        
        # attack profile
        column_list = ['module idx', 'bit-flip idx', 'module name', 'weight idx',
                    'weight before attack', 'weight after attack', 'validation accuracy', 'accuracy drop']
        df.columns = column_list
        df['trial seed'] = id_string(args)
        export_csv = df.to_csv(os.path.join(args.save_path, f'run_{id_string(args)}', 'attack_profile.csv'), index=None)
        

def validate(val_loader, model, criterion, summary_output=False, gpu=0):
    #print(torch.cuda.memory_summary())

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    #device = torch.device(f"cuda:{gpu}")
    #model = model.to(device)

    with suppress():
        # switch to evaluate mode
        model.eval()
        output_summary = [] # init a list for output summary

        with torch.no_grad():
            for i, (input, target) in enumerate(tqdm(val_loader)):
                target = target.cuda(non_blocking=True)
                input = input.cuda()

                # compute output
                output = model(input)
                loss = criterion(output, target)
                
                # summary the output
                if summary_output:
                    tmp_list = output.max(1, keepdim=True)[1].flatten().cpu().numpy() # get the index of the max log-probability
                    output_summary.append(tmp_list)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

            _logger.warning(
                '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
                .format(top1=top1, top5=top5, error1=100 - top1.avg))
        
        torch.cuda.empty_cache()
        if summary_output:
            output_summary = np.asarray(output_summary).flatten()
            return top1.avg, top5.avg, losses.avg, output_summary
        else:
            return top1.avg, top5.avg, losses.avg


if __name__ == '__main__':
    args = parser.parse_args()
    setup_default_logging(log_path=f"{args.save_path}/run_{id_string(args)}/log.txt")

    torch.cuda.empty_cache()
    _logger.warning(' '.join(sys.argv))
    main(args)
