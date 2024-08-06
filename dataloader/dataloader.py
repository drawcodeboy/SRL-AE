from torch.utils.data import Dataset
import os
import wfdb
from matplotlib import pyplot as plt
import pandas as pd

class PTB_XL_Dataset(Dataset):
    def __init__(self, data_dir: str, metadata: pd.DataFrame, mode: str='train'):
        self.metadata = metadata
    
    def _check_and_load(self, mode):
        '''
        (1) load가 되는가
        (2) 내부에 ECG 데이터가 존재하는가
        (3) label은 문제 없는가
        '''
        pass
    
    @staticmethod
    def visualize(data_path):
        if os.path.splitext(data_path)[1] != '':
            raise AssertionError('Extenstion이 존재')
        
        sample = wfdb.rdsamp(data_path)
        
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
            
            axes[i].set_xlim(-0.2, 10.2) # x축 범위 지정 (기본은 너무 넓어 보임).3
            axes[i].set_xticks(major_xticks) # x축 간격 1초씩 나오도록 함.
            axes[i].grid(linestyle='-.', axis='x')
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0., hspace=0.) # Lead 간 간격 붙이기
        plt.show()
        

if __name__ == '__main__':
    sample_path = r"E:\ECG_AD\data\PTB-XL\records100\00000\00154_lr"
    sample = wfdb.rdsamp(sample_path) 
    # sample = Tuple[np.ndarray, Dict]
    # sample[0] = np.ndarray, (sig_len, n_sig)
    # sample[1] = Dict
    
    print(sample[0].shape, [k for k in sample[1].keys()])
    PTB_XL_Dataset.visualize(sample_path)