import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from scipy.io import arff
import os

__all__ = ['ECG5000_Dataset']

class ECG5000_Dataset(Dataset):
    def __init__(self,
                 data_dir: str='data/ECG5000',
                 mode: str='train',
                 train_size: int=2000,
                 val_size: int=100):
        '''
        Args:
            - data_dir: Data의 Root 디렉터리
            - mode: 'train', 'val', 'test'
            - train_size: train set size
            - val_size: val set size
        '''
        self.data_dir = data_dir
        self.mode = mode
        self.train_size = train_size
        self.val_size = val_size
        
        self.len = 140 # all samples length is 140, 141 is label
    
        if self.mode not in ['train', 'val', 'test']:
            raise ValueError('Check your dataset mode')

        self.class_map = {
            "b'1'": 0, # N
            "b'2'": 1, # R-on-T
            "b'3'": 2, # PVC
            "b'4'": 3, # SP
            "b'5'": 4, # UB
        }
        
        self.data_li = []
        
        self._check_and_load()
        
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        sample = self.data_li[idx][0].to_numpy(dtype=np.float32)
        target = self.data_li[idx][1]
        
        # MaxAbs Scaling
        # 신호의 변동성 (-1, 1)을 위해서 다음과 같은 전처리를 수행
        
        sample_abs = np.abs(sample)
        max_abs = np.max(sample_abs)
        
        if max_abs != 0.:
            sample = sample / max_abs
        
        # transform to tensor
        sample = torch.tensor(sample, dtype=torch.float32).view(-1, 1)
        target = torch.tensor(target)
        
        return sample, target
    
    def _check_and_load(self):
        data_train = arff.loadarff(os.path.join(self.data_dir, "ECG5000_TRAIN.arff"))
        df_train = pd.DataFrame(data_train[0])
        
        data_test = arff.loadarff(os.path.join(self.data_dir, "ECG5000_TEST.arff"))
        df_test = pd.DataFrame(data_test[0])
        
        # [b'1' b'2' b'3' b'4' b'5']
        # ['N', 'R-on-T', 'PVC', 'SP', 'UB']
        targets = np.unique(df_train['target'])
        
        df = pd.concat([df_train, df_test], axis=0)
        df_normal = df[df['target'] == targets[0]]
        df_abnormal = df[df['target'] != targets[0]]
        
        df_train = df_normal[:self.train_size]
        df_test = pd.concat((df_normal[self.train_size:], df_abnormal), axis=0)
        
        train_range = self.train_size - self.val_size
        
        if self.mode == 'train':
            for idx in range(0, train_range):
                sample = df_train.iloc[idx][:self.len]
                label = self.class_map[str(df_train['target'].iloc[idx])]
                self.data_li.append([sample, label])
                
        elif self.mode == 'val':
            for idx in range(train_range, len(df_train)):
                sample = df_train.iloc[idx][:self.len]
                label = self.class_map[str(df_train['target'].iloc[idx])]
                self.data_li.append([sample, label])
                
        elif self.mode == 'test':
            for idx in range(0, len(df_test)):
                sample = df_test.iloc[idx][:self.len]
                label = self.class_map[str(df_test['target'].iloc[idx])]
                self.data_li.append([sample, label])