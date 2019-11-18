import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import os
import time
import datetime
import yaml
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython import embed

import models
from dataset import HFDataset, transform, data_prefetcher
from data_preparing import name2index
from utils import loss, AverageMeter
from utils import calc_f1, LRScheduler


# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# Parse Hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/ResNeXt50_2x16d.yaml')
parser.add_argument('-v', '--val', dest='val', action='store_true',
                    help='evaluate model on validation set')  # validation
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
args = parser.parse_args()


def main():
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load config
    print(args.config, flush=True)
    with open(args.config, 'r') as f:
        opt = yaml.safe_load(f)
    for k, v in opt.items():
        print('{} : {}'.format(k, v), flush=True)
        setattr(args, k, v)
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    # Train
    if not args.evaluate and not args.val:
        # Dataset
        train_dataset = HFDataset(args.data_root, args.train_csv_path,
                                  args.data_lens, train=True)
        val_dataset = HFDataset(args.data_root, args.validation_csv_path,
                                args.data_lens, train=False)
        # Data loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=32,
                                                 shuffle=False,
                                                 num_workers=1)
        # Model
        model = getattr(models, args.model)(args.num_classes).to(device)
        if args.load_model_path:
            model.load(args.load_model_path)

        # Loss and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr,
                                     weight_decay=args.weight_decay)
        if args.loss == 'weighted_binary_crossentropy':
            weight = train_dataset.weight
            weight = torch.Tensor(weight).unsqueeze(0).to(device)
            criterion = getattr(loss, args.loss)(weight)
        else:
            criterion = getattr(loss, args.loss)

        niters = len(train_loader)
        lr_scheduler = LRScheduler(optimizer, niters, args)

        # Run
        train(train_loader, val_loader, model, optimizer,
              criterion, lr_scheduler, device, args)
    # Val
    elif args.val:
        val_dataset = HFDataset(args.data_root, args.validation_csv_path,
                                args.data_lens, train=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1)
        model = getattr(models, args.model)(args.num_classes).to(device)
        assert args.load_model_path
        model.load(args.load_model_path)
        acc_val = val(val_loader, model, device, flip=args.flip)
        print('Validation Accuracy: {} %'.format(acc_val), flush=True)
    # Test
    elif args.evaluate:
        model = getattr(models, args.model)(args.num_classes).to(device)
        assert args.load_model_path
        model.load(args.load_model_path)
        test(model, args, device)
    return


def train(train_loader, val_loader, model, optimizer, criterion, lr_scheduler, device, opt):
    '''
    model training
    :param: train_loader: dataloader, val_loader: dataloader, model: cpkt,
    optimizer: optimizer, criterion: weighted_binary_crossentropy, lr_scheduler: LRScheduler,
    device: device, opt: dict
    :return
    '''
    total_step = len(train_loader)
    best_acc = -1
    losses = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()
    print_freq = 20
    iter_per_epoch = len(train_loader)
    iter_sum = iter_per_epoch * opt.epochs
    fast_train = hasattr(opt, 'fast_train')
    writer = SummaryWriter(opt.model_save_path)

    for epoch in range(opt.epochs):
        model.train()

        prefetcher = data_prefetcher(train_loader)
        datas, ages, sexs, labels = prefetcher.next()
        i = 0
        while datas is not None:
            i += 1
            lr_scheduler.update(i, epoch)
        # for i, (datas, ages, sexs, labels) in enumerate(train_loader):
            datas = datas.to(device)
            ages = ages.to(device)
            sexs = sexs.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(datas, ages, sexs)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update tensorboard
            batch_time.update(time.time() - end)
            losses.update(loss.item(), datas.size(0))
            # Print
            if (i + 1) % print_freq == 0:
                iter_used = epoch * iter_per_epoch + i
                used_time = batch_time.sum
                total_time = used_time / iter_used * iter_sum
                used_time = str(datetime.timedelta(seconds=used_time))
                total_time = str(datetime.timedelta(seconds=total_time))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, LR:{:.5f}, Time[{:.7s}/{:.7s}]'
                      .format(epoch + 1, opt.epochs, i + 1, total_step, loss.item(),
                              optimizer.param_groups[0]['lr'], used_time, total_time), flush=True)
                writer.add_scalar(
                    'Learning_rate', optimizer.param_groups[0]['lr'], iter_used)
                writer.add_scalar('Train/Avg_Loss', losses.avg, iter_used)
            end = time.time()
            datas, ages, sexs, labels = prefetcher.next()

        if not fast_train:
            # acc in train set
            acc_train = val(train_loader, model, device)
            print('Train Accuracy: {} %'.format(acc_train), flush=True)
            writer.add_scalar('Train/F1_Score', acc_train, iter_used)
            # acc in validation set
            acc_val = val(val_loader, model, device)
            if acc_val > best_acc:
                # Save the model checkpoint
                best_acc = acc_val
                if epoch > int(opt.epochs * 0.8):
                    save_name = args.model + '_e{}.ckpt'.format(epoch)
                    save_path = opt.model_save_path + save_name
                    torch.save(model.state_dict(), save_path)
            print('Validation Accuracy: {} %'.format(acc_val), flush=True)
            writer.add_scalar('Validation/F1_Score', acc_val, iter_used)
        else:
            if epoch > int(opt.epochs * 0.8):
                acc_val = val(val_loader, model, device)
                if acc_val > best_acc:
                    best_acc = acc_val
                    save_name = args.model + '_e{}.ckpt'.format(epoch)
                    save_path = opt.model_save_path + save_name
                    torch.save(model.state_dict(), save_path)
    return


def val(val_loader, model, device, flip=False):
    '''
    model evaluation
    :param: val_loader: dataloader, model: cpkt, device:device, flip: bool
    :return: float
    '''
    model.eval()
    with torch.no_grad():
        score = []
        for datas, ages, sexs, labels in val_loader:
            datas = datas.to(device)
            ages = ages.to(device)
            sexs = sexs.to(device)
            labels = labels.to(device)
            outputs = model(datas, ages, sexs)
            if flip:
                datas_flip = datas.flip([2])
                outputs_flip = model(datas_flip)
                outputs_mean = torch.add(outputs, outputs_flip) / 2
                x = calc_f1(labels, outputs_mean)
            else:
                x = calc_f1(labels, outputs)
            score.append(x)
        test_acc = sum(score) / len(score)
    return test_acc


def test(model, opt, device):
    '''
    generate predictions
    :param: model: cpkt, opt: dict, device:device
    :return 
    '''
    model.eval()
    name2idx = name2index(opt.arrythmia_path)
    idx2name = {idx: name for name, idx in name2idx.items()}
    sub_dir = './submit'
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    sub_file = '%s/subB_%s.txt' % (sub_dir, '1')
    with open(sub_file, 'w', encoding='utf-8') as fout:
        with torch.no_grad():
            with open(opt.test_label, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                fout.write(line.strip('\n'))
                id = line.split('\t')[0]
                file_path = os.path.join(opt.test_root, id)
                df = pd.read_csv(file_path, sep=' ').values
                data = transform(df).unsqueeze(0)
                data = data.to(device)
                output = model(data).squeeze().cpu().numpy()
                ixs = [i for i, out in enumerate(output) if out > 0.5]
                for i in ixs:
                    fout.write("\t" + idx2name[i])
                fout.write('\n')
    return


if __name__ == '__main__':
    main()
