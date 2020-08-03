import glob
import pretty_midi
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
# from sklearn.preprocessing import StandardScaler
import pickle
import os
from midi import *
import logging
import sys
from log import LOG

class MidiDataset(Dataset):
    def __init__(self, dirlist):
        datasets = []
        labels = []
        for dir, label, type, usecache in dirlist:
            dataset = self.get_dataset(dir, type, usecache)
            datasets.append(dataset)
            num = dataset.shape[0]
            labels.append(np.full([num], label, dtype=np.int64))
            LOG.info(f'Dataset loaded. label:{label}, type:{type:<6}, use cache:{str(usecache):<5}, num:{num:<5}, dir:"{dir}"' )
        datas = np.vstack(datasets)

        LOG.info('Preprcessing datasets...')
        datas = self.preprocess(datas)
        labels = np.hstack(labels)

        self.midi_datas = torch.from_numpy(datas).float()
        self.labels = torch.from_numpy(labels)

    def get_dataset(self, dir, type, usecache = True):
        plkpath = os.path.join(dir, 'cache.plk')
        if usecache and os.path.exists(plkpath):
            with open(plkpath, 'rb') as f:
                dataset = pickle.load(f)
        else:
            if type == 'csmt':
                dataset = self.get_csmt_dataset(dir)
            elif type == 'online':
                dataset = self.get_online_dataset(dir)
            else:
                raise Exception('类型错误')
            with open(plkpath, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        return dataset

    def get_csmt_dataset(self, dir):
        return np.vstack([load_csmt_midi(os.path.join(dir, filename)) for filename in os.listdir(dir) if filename.endswith('.mid')])

    def get_online_dataset(self, dir):
        LOG.info(f'start process online dataset "{dir}"')
        dataset = []
        files = list(os.listdir(dir))
        cnt, n = 0, len(files)
        LOG.info(f'find {n} files!')
        cnt, n = 0, len(files)
        for i, filename in enumerate(files):
            filename = os.path.join(dir, filename)
            if filename.endswith('.mid'):
                try:
                    data = load_online_midi(filename)
                    if data is not None:
                        dataset.append(data)
                        cnt += len(data)
                except:
                    pass
                sys.stdout.write(f'\r{i+1}/{n} num:{cnt}')
            if cnt > 6000:
                break
        sys.stdout.write('\r')
        return np.vstack(dataset)

    def preprocess(self, datas):
        return np.vstack([self.vec2mat(vec) for vec in datas]).reshape(datas.shape[0], 1, 128, 128)

    def vec2mat(self, vec):
        convert = [0, 4, 4, 2, 2, 1, 6, 1, 3, 3, 5, 5]
        #     convert = [0, 7, 8, 3, 4, 1, 12, 2, 5, 6, 10, 11]
        #     convert = [0,5,7, 3,4, 8, 9, 1, 2, 10, 11, 6]
        # return np.array([[12 - (b - a) % 12 for b in vec] for a in vec])
        return np.array([[b - a for b in vec] for a in vec])


    def __len__(self):
        return len(self.midi_datas)

    def __getitem__(self, index):
        return self.midi_datas[index], self.labels[index]



if __name__ == '__main__':
    dirlist = [
        [r'..\baseline\data\eval\fake', 0, 'csmt'],
        [r'..\baseline\data\eval\real', 1, 'csmt'],
    ]
    dataset = MidiDataset(dirlist)
    for data, label in dataset:
        print(data.shape, label)