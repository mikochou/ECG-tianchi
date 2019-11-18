from tqdm import tqdm
import pandas as pd
import numpy as np
from IPython import embed
np.random.seed(0)


def get_arrythmias(arrythmias_path):
    '''
    read labels
    :param: file path
    :return: list
    '''
    with open(arrythmias_path, "r") as f:
        data = f.readlines()
    arrythmias = [d.strip() for d in data]
    print(len(arrythmias))
    return arrythmias


def get_dict(arrythmias_path):
    '''
    build a dictionary for conversion
    :param path: file path
    :return: dict
    '''
    arrythmias = get_arrythmias(arrythmias_path)
    str2ids = {}
    id2strs = {}
    for i, a in enumerate(arrythmias):
        str2ids[a] = i
        id2strs[i] = a
    return str2ids, id2strs


def get_train_label(label_path, str2ids, train_csv_path, validation_csv_path,
                    trainval_csv_path, train_len):
    '''
    get train label
    :param path: file path, int
    '''
    with open(label_path, "r", encoding='UTF-8') as f:
        data = f.readlines()
    labels = [d.strip() for d in data]
    label_dicts = {}
    label_dicts["index"] = []
    label_dicts["age"] = []
    label_dicts["sex"] = []
    label_dicts["one_label"] = []
    i = 0
    for l in tqdm(labels):
        i += 1
        ls = l.split("\t")
        if len(ls) <= 1:
            continue
        label_dicts["index"].append(ls[0])
        label_dicts["age"].append(ls[1])
        label_dicts["sex"].append(ls[2])
        one_label = np.zeros(len(str2ids),)
        for ls1 in ls[3:]:
            one_label[str2ids[ls1]] = 1
        label_dicts["one_label"].append(list(one_label))

    df = pd.DataFrame(label_dicts)
    df = df.sample(frac=1)
    df.to_csv(trainval_csv_path, index=None)
    # df_train = df[:train_len] //划分训练集
    print(df.shape)
    #df_val = df[train_len:]
    #df_train.to_csv(train_csv_path, index=None)
    #df_val.to_csv(validation_csv_path, index=None)


def name2index(path):
    '''
    Convert label name to index
    :param path: file path
    :return: dict
    '''
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx


if __name__ == '__main__':
    train_len = 16000
    arrythmia_path = '../data/hf_round2_arrythmia.txt'
    #label_path = '../user_data/hf_round1_label.txt'
    #label_path = '../user_data/heifei_round1_ansA_20191008.txt'
    label_path = '../user_data/hf_round2_train.txt'
    #train_csv_path = '../user_data/hf_round1_label_train.csv'
    train_csv_path = '../user_data/hf_round2_label_train.csv'
    validation_csv_path = '../user_data/hf_round2_label_validation.csv'
    trainval_csv_path = '../user_data/hf_round2_label_testA.csv'

    str2ids, id2strs = get_dict(arrythmia_path)
    get_train_label(label_path, str2ids, train_csv_path,
                    validation_csv_path, trainval_csv_path, train_len)
