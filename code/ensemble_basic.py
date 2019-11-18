import torch
import torch.nn as nn

import os
import yaml
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython import embed

import models
from dataset import HFDataset, transform, transfer_age, transfer_sex
from data_preparing import name2index
from utils import binary_crossentropy as criterion
from utils import calc_f1, LRScheduler


# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# Parse Hyper parameters
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/ensemble_dev_testB.yaml')
parser.add_argument('-v', '--val', dest='val', action='store_true',
                    help='evaluate model on validation set')
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

    # Load model
    assert len(args.model_name) == len(
        args.model_path) == len(args.model_weight)
    model_list = []
    for i in range(len(args.model_name)):
        name = args.model_name[i]
        path = args.model_path[i]
        model = getattr(models, name)(args.num_classes).to(device)
        model.load(path)
        model.eval()
        model_list.append(model)

    model_weight = np.array(args.model_weight)
    model_weight = model_weight / np.sum(model_weight)
    # Val
    if args.val:
        val_dataset = HFDataset(args.data_root, args.validation_csv_path,
                                args.data_lens, train=False)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=1)
        val(val_loader, model_list, model_weight, device, flip=args.flip)
    # Test
    elif args.evaluate:
        test(model_list, model_weight, args, device)
    return


def val(val_loader, model_list, model_weight, device, flip=False):
    '''
    model evaluation
    :param: val_loader: dataloader, model_list: list, model_weight: list, device:device, flip: bool
    :return: float
    '''
    model_weight = torch.Tensor(model_weight).to(device)
    with torch.no_grad():
        score_list = []
        for datas, ages, sexs, labels in tqdm(val_loader):
            datas = datas.to(device)
            #ages = ages.to(device)
            #sexs = sexs.to(device)
            labels = labels.to(device)
            output_list = []
            for index, model in enumerate(model_list):
                outputs = model(datas)
                if flip:
                    datas_flip = datas.flip([2])
                    outputs_flip = model(datas_flip, ages, sexs)
                    outputs_mean = torch.add(outputs, outputs_flip) / 2
                    outputs = outputs_mean
                weight = model_weight[index]
                outputs *= weight
                output_list.append(outputs)
            outputs_mean = torch.sum(torch.stack(output_list), dim=0)
            score = calc_f1(labels, outputs_mean)
            score_list.append(score)
        test_acc = sum(score_list) / len(score_list)
        print('Test Accuracy: {}/{}={} %'.format(sum(score_list),
                                                 len(score_list), test_acc), flush=True)
    return test_acc


def test(model_list, model_weight, opt, device):
    '''
    generate predictions
    :param: model_list: list, model_weight: list, opt: dict, mname:list, device:device
    :return 
    '''
    model_weight = torch.Tensor(model_weight).to(device)
    name2idx = name2index(opt.arrythmia_path)
    idx2name = {idx: name for name, idx in name2idx.items()}
    sub_dir = '../prediction_result'
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    sub_file = '%s/result_%s.txt' % (sub_dir, '1')
    with open(sub_file, 'w', encoding='utf-8') as fout:
        with torch.no_grad():
            with open(opt.test_label, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in tqdm(lines):
                fout.write(line.strip('\n'))
                ind, age, sex = line[:-1].split('\t')
                file_path = os.path.join(opt.test_root, ind)
                df = pd.read_csv(file_path, sep=' ').values
                data = transform(df).unsqueeze(0)
                data = data.to(device)

                age = transfer_age(age)
                age = torch.Tensor(age).unsqueeze(0).to(device)
                sex = transfer_sex(sex)
                sex = torch.FloatTensor([sex]).to(device)
                output_list = []
                for index, model in enumerate(model_list):
                    outputs = model(data, age, sex)
                    weight = model_weight[index]
                    outputs *= weight
                    output_list.append(outputs)
                outputs_mean = torch.sum(torch.stack(output_list), dim=0)
                output = outputs_mean.squeeze().cpu().numpy()

                ixs = [i for i, out in enumerate(output) if out > 0.5]
                for i in ixs:
                    fout.write("\t" + idx2name[i])
                fout.write('\n')
    return


if __name__ == '__main__':
    main()
