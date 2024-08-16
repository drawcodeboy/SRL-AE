from typing import Optional
from .lstm_ae import LSTMAutoencoder

def load_model(model_name:str='LSTM-AE', 
               num_attn_heads:Optional[int]=None):
    
    if model_name == 'LSTM-AE':
        return LSTMAutoencoder(num_layers=5,
                               in_out_dim=12, # 12 ECG leads
                               hidden_dims=[128, 64, 32, 16, 8])
        
    elif model_name == 'ResLSTM-AE':
        return LSTMAutoencoder(num_layers=5,
                               in_out_dim=12, # 12 ECG leads
                               hidden_dims=[128, 64, 32, 16, 8],
                               encoder_lstm_type='res',
                               decoder_lstm_type='res')
        
    elif model_name == 'SKIP-LSTM-AE':
        return LSTMAutoencoder(num_layers=5,
                               in_out_dim=12, # 12 ECG leads
                               hidden_dims=[128, 64, 32, 16, 8],
                               connect='skip')
    
    elif model_name == 'CA-LSTM-AE':
        return LSTMAutoencoder(num_layers=5,
                               in_out_dim=12, # 12 ECG leads
                               hidden_dims=[128, 64, 32, 16, 8],
                               connect='cross-att',
                               num_attn_heads=num_attn_heads)
    
    elif model_name == 'CA-ResLSTM-AE' or model_name == 'CarlaNet':
        return LSTMAutoencoder(num_layers=5,
                               in_out_dim=12, # 12 ECG leads
                               hidden_dims=[128, 64, 32, 16, 8],
                               encoder_lstm_type='res',
                               decoder_lstm_type='res',
                               connect='cross-att',
                               num_attn_heads=num_attn_heads)
        
    else:
        raise AssertionError(f"{model_name} is not supported.")