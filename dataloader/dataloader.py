import torch
from torch.utils.data import Dataset

import os, sys, time
import pandas as pd
import numpy as np

import wfdb
from wfdb.processing import normalize_bound
import json
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, lfilter

from typing import Dict

class PTB_XL_Dataset(Dataset):
    def __init__(self, data_dir: str, metadata_path: pd.DataFrame, mode: str='train'):
        '''
        Args:
            - data_dir: record100 or record500 직전의 부모 디렉터리의 위치
            - metadata_path: ptbxl_database.csv의 위치
            - mode: train/test
        '''
        self.data_dir = data_dir
        self.metadata = pd.read_csv(metadata_path)
        self.mode = mode
        
        if self.mode not in ['train', 'test']:
            raise ValueError('Check your dataset mode')
        
        self.data_li = [] # [file_path, label]
        self.class_map = {
            'NORM': 0, # normal ECG
            'NDT': 1, # non-diagnostic T abnormalities
            'NST_': 2, # non-specific ST changes
            'DIG': 3, # digitalis-effect
            'LNGQT': 4 # long QT-interval
        }
        self.abnormal_cnt = 0
        
        self._check_and_load()
        self._mode_transform()
    
    def __len__(self):
        return len(self.data_li)
    
    def __getitem__(self, idx):
        # (1) 0.5Hz ~ 40Hz Bandpass filtering
        # (2) 0-1 Normalization
        
        sample = wfdb.rdsamp(self.data_li[idx][0])
        target = self.data_li[idx][1]
        
        sample = PTB_XL_Dataset.preprocess(sample)
        
        # target은 tensor로 변환 될 필요 X
        # input tensor(ECG) shape = (seq_len, dim)
        ecg_record = torch.tensor(sample[0], dtype=torch.float32)
        
        return ecg_record, target
    
    @staticmethod
    def preprocess(sample):
        ecg_record = []
        for channel in range(sample[0].shape[1]): # 12 channels
            ecg_lead = sample[0][:, channel]
            
            # Bandpass filtering
            ecg_lead = PTB_XL_Dataset.butter_bandpass_filter(ecg_lead, 
                                                             0.5, 40, sample[1]['fs'])
            
            # 0-1 Normalization
            ecg_lead = normalize_bound(ecg_lead, 0., 1.)
            
            ecg_record.append(ecg_lead)
            
        # 위 과정을 수행하면서 shape이 transpose로 바뀌었음
        sample = (np.array(ecg_record, dtype=np.float32).T, sample[1])
        
        return sample
    
    @staticmethod
    def butter_bandpass(lowcut, highcut, fs, order):
        '''
        Args:
            lowcut
            highcut
            fs: Sampling Rate
            ordrer: 깎는 정도, 높을수록 급격하게 깎아냄.
        '''
        nyq = 0.5 * fs # 나이퀴스트 주파수: fs의 절반
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass') # 필터 계수 구함
        return b, a
    
    @staticmethod
    def butter_bandpass_filter(sample, lowcut, highcut, fs, order=5):
        b, a = PTB_XL_Dataset.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, sample) # 주어진 필터 계수를 통해 필터링
        return y
    
    def _mode_transform(self):
        # Normal Data들이 맨 앞에 오게 정렬
        self.data_li.sort(key=lambda x: x[1]) # inplace-sort
        
        # Train 데이터의 수 구하기
        train_cnt = len(self.data_li) - 2*self.abnormal_cnt
        
        if self.mode == 'train':
            self.data_li = self.data_li[:train_cnt]
        elif self.mode == 'test':
            self.data_li = self.data_li[train_cnt:]
    
    def _check_pure_likelihood(self, label_dict: Dict):
        # Sample의 Label Dictionary를 통해서 순수한 Label을 가진
        # Sample인지를 판정, 판정 방식은 특정 label에 대해 likelihood가 100을 갖고
        # 나머지는 0을 갖도록 함.
        for label in self.class_map.keys():
            purity = False
            for k, v in label_dict.items():
                if (k == label and v == 100):
                    purity = True
                elif (k != label and v == 0):
                    continue
                else:
                    purity = False
                    break
            if purity == True:
                return label
        return 'NO LABEL'
    
    def _check_and_load(self):
        '''
        (1) load가 되는가
        (2) 내부에 ECG 데이터가 존재하는가
        (3) label은 문제 없는가(순수한가)
        '''
        start_time = time.time()
        for idx, (file_path, target_dict) in enumerate(zip(self.metadata['filename_lr'], self.metadata['scp_codes'])):
            print(f"\rCheck & Load data: {100*idx/len(self.metadata):.2f}%", end='')
            
            file_path = os.path.join(self.data_dir, file_path)
            
            # Check Target (Pure(likelihood 100%) Check)
            target_dict = target_dict.replace('\'', '\"') # Python syntax -> JSON syntax
            target_dict = json.loads(target_dict)
            
            target = self._check_pure_likelihood(target_dict)
            if target in list(self.class_map.keys())[1:]: # Abnormal Label이라면
                self.abnormal_cnt += 1
            elif target == 'NO LABEL':
                continue
            
            target = self.class_map[target]
            
            # Load
            # Target 먼저 Check하는 것이 훨씬 빠르다.
            try:
                sample = wfdb.rdsamp(file_path)
            except:
                print(f"\n{file_path[-8:-3]} cannot load by wfdb.rdsamp")
                continue
            
            # Check Data (NaN)
            if np.isnan(sample[0]).any():
                print(f"\n{file_path[-8:-3]} got missing value(NaN).")
                continue
            
            self.data_li.append([file_path, target])
            
        load_time = int(time.time() - start_time)
        print(f"\nCheck & Load data time: {load_time//60}m {load_time%60}s")
    
    @staticmethod
    def visualize(data_path, preprocess=False):
        if os.path.splitext(data_path)[1] != '':
            raise AssertionError('Extenstion이 존재')
        
        sample = wfdb.rdsamp(data_path)
        
        if preprocess == True:
            plt.figure(2) # 여러 창 띄울 때도 있음. original은 Figure 1, preprocess는 Figure 2
            sample = PTB_XL_Dataset.preprocess(sample)
        
        fig, axes = wfdb.plot.plot_items(signal=sample[0],
                                         title=f'PTB-XL Record {data_path[-8:-3]}',
                                         # sig_name=sample[1]['sig_name'],
                                         # sig_units=sample[1]['units'],
                                         fs=sample[1]['fs'],
                                         time_units='seconds',
                                         sig_style=['r-'], # Format String 부분 참고, https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
                                         figsize=(12, 10),
                                         return_fig_axes=True)
        
        # Grid
        major_xticks = [i for i in range(0, 11)]
        
        for i in range(len(axes)): 
            axes[i].patch.set_facecolor('#fffced')
            axes[i].tick_params(right=True, left=False,
                                labelright=True, labelleft=False) # y축의 간격과 이에 대한 범위를 오른쪽으로 옮김
            
            axes[i].set_ylabel(f"{sample[1]['sig_name'][i]}\n({sample[1]['units'][i]})", rotation=0, labelpad=20, loc='center')
            axes[i].yaxis.set_label_coords(-0.03, 0.3)  # Ylabel의 위치 x, y 좌표 설정 (기본값은 (0, 0.5)), 위 함수 loc에서 center로 조절해도 미세하게 안 맞음
            
            axes[i].set_xlim(-0.1, 10.1) # x축 범위 지정 (기본은 너무 넓어 보임).3
            axes[i].set_xticks(major_xticks) # x축 간격 1초씩 나오도록 함.
            axes[i].grid(linestyle='-', which='major', axis='x', linewidth=1.2)
            
            # Minor Grid
            axes[i].xaxis.set_minor_locator(MultipleLocator(0.2))
            axes[i].grid(linestyle='-.', which='minor', axis='x')
            
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0., hspace=0.) # Lead 간 간격 붙이기
        

if __name__ == '__main__':
    # Train, Test 실험
    '''
    train_ds = PTB_XL_Dataset(data_dir='data/PTB-XL',
                              metadata_path='data/PTB-XL/ptbxl_database.csv',
                              mode='train')
    
    test_ds = PTB_XL_Dataset(data_dir='data/PTB-XL',
                             metadata_path='data/PTB-XL/ptbxl_database.csv',
                             mode='test')
    
    print(len(train_ds), len(test_ds))
    '''
    
    train_ds = PTB_XL_Dataset(data_dir='data/PTB-XL',
                              metadata_path='data/PTB-XL/ptbxl_database.csv',
                              mode='train')
    ecg, target = train_ds[0]
    print(ecg.shape, target)
    
    # 시각화 실험
    sample_path = r"E:\ECG_AD\data\PTB-XL\records100\00000\00154_lr"
    
    PTB_XL_Dataset.visualize(sample_path, preprocess=False)
    PTB_XL_Dataset.visualize(sample_path, preprocess=True)
    plt.show()