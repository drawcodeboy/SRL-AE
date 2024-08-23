import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
from scipy.io import arff
from matplotlib import pyplot as plt
import os, sys

__all__ = ['ECG5000_Dataset']

class ECG5000_Dataset(Dataset):
    def __init__(self,
                 data_dir: str='data/ECG5000',
                 mode: str='train',
                 val_size: float=0.2):
        '''
        Args:
            - data_dir: Data의 Root 디렉터리
            - mode: 'train', 'val', 'test'
        '''
        self.data_dir = data_dir
        self.mode = mode
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
        
        # train set(includes val set)에서 abnormal이 있을 필요는 없음.
        # train set의 abnormal을 test set으로 옮김
        df_train_abnormal = df_train[df_train['target'] != targets[0]]
        df_test = pd.concat([df_test, df_train_abnormal], axis=0) # test set에 추가
        df_train.drop(df_train[df_train['target'] != targets[0]].index, inplace=True) # train set에서 abnormal 제거
        
        train_range = int(len(df_train) * (1-self.val_size))
        
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


if __name__ == '__main__':
    train_ds = ECG5000_Dataset(mode='train')
    val_ds = ECG5000_Dataset(mode='val')
    test_ds = ECG5000_Dataset(mode='test')
    print(len(train_ds), len(val_ds), len(test_ds))
    
    sample = train_ds[0][0].detach().cpu().numpy()
    print(sample.shape)
    print(train_ds[0][1])
    
    
    plt.plot(np.linspace(0, 100, len(sample)), sample)
    plt.show()