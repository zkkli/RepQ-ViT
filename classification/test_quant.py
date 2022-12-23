import argparse
import time
import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np

from utils import *
from quant import *


def get_args_parser():
    parser = argparse.ArgumentParser(description="RepQ-ViT", add_help=False)
    parser.add_argument("--model", default="deit_small",
                        choices=['vit_small', 'vit_base',
                            'deit_tiny', 'deit_small', 'deit_base', 
                            'swin_tiny', 'swin_small'],
                        help="model")
    parser.add_argument('--dataset', default="/dataset/imagenet/",
                        help='path to dataset')
    parser.add_argument("--calib-batchsize", default=32,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--val-batchsize", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=16, type=int,
                        help="number of data loading workers (default: 16)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    parser.add_argument("--print-freq", default=100,
                        type=int, help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")

    parser.add_argument('--w_bits', default=4,
                        type=int, help='bit-precision of weights')
    parser.add_argument('--a_bits', default=4,
                        type=int, help='bit-precision of activation')

    return parser


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    print(args)
    seed(args.seed)

    model_zoo = {
        'vit_small' : 'vit_small_patch16_224',
        'vit_base' : 'vit_base_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
    }
    
    device = torch.device(args.device)
    
    # Build dataloader
    print('Building dataloader ...')
    train_loader, val_loader = build_dataset(args)
    for data, target in train_loader:
        calib_data = data.to(device)
        break
    calib_data.to(device)

    # Build model
    print('Building model ...')
    model = build_model(model_zoo[args.model])
    model.to(device)
    model.eval()

    wq_params = {'n_bits': args.w_bits, 'channel_wise': True}
    aq_params = {'n_bits': args.a_bits, 'channel_wise': False}
    q_model = quant_model(model, input_quant_params=aq_params, weight_quant_params=wq_params)
    q_model.to(device)
    q_model.eval()

    # Initial quantization
    print('Performing initial quantization ...')
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_model(calib_data)

    # Scale reparameterization
    print('Performing scale reparameterization ...')
    with torch.no_grad():
        module_dict={}
        q_model_slice = q_model.layers if 'swin' in args.model else q_model.blocks
        for name, module in q_model_slice.named_modules():
            module_dict[name] = module
            idx = name.rfind('.')
            if idx == -1:
                idx = 0
            father_name = name[:idx]
            if father_name in module_dict:
                father_module = module_dict[father_name]
            else:
                raise RuntimeError(f"father module {father_name} not found")

            if 'norm1' in name or 'norm2' in name or 'norm' in name:
                if 'norm1' in name:
                    next_module = father_module.attn.qkv
                elif 'norm2' in name:
                    next_module = father_module.mlp.fc1
                else:
                    next_module = father_module.reduction
                
                act_delta = next_module.input_quantizer.delta.reshape(-1)
                act_zero_point = next_module.input_quantizer.zero_point.reshape(-1)
                act_min = -act_zero_point * act_delta
                
                target_delta = torch.mean(act_delta)
                target_zero_point = torch.mean(act_zero_point)
                target_min = -target_zero_point * target_delta

                r = act_delta / target_delta
                b = act_min / r - target_min

                module.weight.data = module.weight.data / r
                module.bias.data = module.bias.data / r - b

                next_module.weight.data = next_module.weight.data * r
                if next_module.bias is not None:
                    next_module.bias.data = next_module.bias.data + torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)
                else:
                    next_module.bias = Parameter(torch.Tensor(next_module.out_features))
                    next_module.bias.data = torch.mm(next_module.weight.data, b.reshape(-1,1)).reshape(-1)

                next_module.input_quantizer.channel_wise = False
                next_module.input_quantizer.delta = target_delta
                next_module.input_quantizer.zero_point = target_zero_point
                next_module.weight_quantizer.inited = False

    # Re-calibration
    set_quant_state(q_model, input_quant=True, weight_quant=True)
    with torch.no_grad():
        _ = q_model(calib_data)

    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # Validate the quantized model
    print("Validating ...")
    val_loss, val_prec1, val_prec5 = validate(
        args, val_loader, q_model, criterion, device
    )


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        target = target.to(device)
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )
    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser('RepQ-ViT', parents=[get_args_parser()])
    args = parser.parse_args()
    main()