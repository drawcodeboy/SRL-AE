from typing import Optional

from .ptb_xl_dataloader import *
from .ecg5000_dataloader import *

def load_dataset(dataset: str="ECG5000",
                 mode: str='train',
                 data_dir: str='data/ECG5000',
                 metadata_path: Optional[str]=None,
                 freq: Optional[int]=None, 
                 seconds: Optional[int]=None,
                 preprocessing: bool=False):
    if dataset=="ECG5000":
        return ECG5000_Dataset(data_dir=data_dir,
                               mode=mode,
                               preprocessing=preprocessing)
    elif dataset=="PTB-XL":
        return PTB_XL_Dataset(data_dir=data_dir,
                              metadata_path=metadata_path,
                              mode=mode,
                              freq=freq,
                              seconds=seconds)