import torch
from torch.utils.data.dataset import Dataset
import os
import math
import pandas as pd
import numpy as np
from IPython import embed
np.random.seed(0)


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, 
                                     size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def verflip(sig):
    return sig[::-1, :]


def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5:
            sig = scaling(sig)
        if np.random.randn() > 0.5:
            sig = verflip(sig)
        if np.random.randn() > 0.5:
            sig = shift(sig)
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


def transfer_age(inp):
    if type(inp) == str:
        if len(inp) > 0:
            inp = float(inp)
        else:
            inp = math.nan
    temp = np.zeros(10)
    if math.isnan( inp ):
        temp[0] = 1
    else:
        n = int(inp // 10)
        if n >= 8:
            n = 8
        temp[n + 1] = 1
    return temp


def transfer_sex(inp):
    if inp == 'FEMALE':
        return 1
    elif inp == 'MALE':
        return 2
    else:
        return 0


class HFDataset(Dataset):
    def __init__(self, data_root, csv_path, data_lens, train=False):
        self.data_root = data_root
        df = pd.read_csv(csv_path)
        age = df['age'].map(transfer_age).values
        age = np.concatenate(age, 0).reshape(-1, 10)
        self.age = torch.Tensor(age)
        sex = df['sex'].map(transfer_sex).values
        self.sex = torch.Tensor(sex)
        df["one_label"] = df["one_label"].map(lambda x: eval(x))
        data_df = pd.DataFrame()
        data_df["index"], data_df["one_label"] = df["index"], df["one_label"]
        self.data = data_df.values
        self.data_lens = data_lens
        self.train = train
        label = np.asarray([np.array(l) for l in self.data[:, 1]])
        self.class_num = label.shape[1]
        # class aware sampling
        self.cas_dic = {}
        for i in range(self.class_num):
            self.cas_dic[i] = np.where(label[:, i] == 1)[0]
        # weighted loss
        weight = 1. / np.log( np.sum(label, axis=0)  + 1e-5)
        #self.weight = weight
        scale_ratio = 1.0 / np.min(weight)
        self.weight = weight * scale_ratio
        del label, age, sex

    def __getitem__(self, index):
        # class aware sampling
        if False:
            ind = index % self.class_num
            index = np.random.permutation(self.cas_dic[ind])[0]
        file_name, onehot_label = self.data[index]
        file_path = os.path.join(self.data_root, file_name)
        df = pd.read_csv(file_path, sep=" ")
        data = df.values[:self.data_lens]
        data = transform(data, train=self.train)
        age = self.age[index]
        sex = self.sex[index]
        onehot_label = np.array(onehot_label)
        onehot_label = torch.Tensor(onehot_label)
        return data, age, sex, onehot_label

    def __len__(self):
        return len(self.data)


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_age, self.next_sex, self.next_label = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_age = None
            self.next_sex = None
            self.next_label = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.cuda(non_blocking=True)
            self.next_age = self.next_age.cuda(non_blocking=True)
            self.next_sex = self.next_sex.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_data = self.next_data.float()
            self.next_age = self.next_age.float()
            self.next_sex = self.next_sex.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        age = self.next_age
        sex = self.next_sex
        label = self.next_label
        if data is not None:
            data.record_stream(torch.cuda.current_stream())
        if age is not None:
            age.record_stream(torch.cuda.current_stream())
        if sex is not None:
            sex.record_stream(torch.cuda.current_stream())
        if label is not None:
            label.record_stream(torch.cuda.current_stream())
        self.preload()
        return data, age, sex, label

