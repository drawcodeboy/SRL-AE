from torchinfo import summary
import torch
from models import load_model

model_1 = load_model('LSTM-AE')
model_2 = load_model('ResLSTM-AE')
model_3 = load_model('SKIP-LSTM-AE')
model_4 = load_model('CA-LSTM-AE', num_attn_heads=1)
model_5 = load_model('CarlaNet', num_attn_heads=1)

# print(model_1)
# print(model_2)

sample = torch.randn(8, 1000, 12)

outputs = model_1(sample)
print(outputs.shape)

outputs = model_2(sample)
print(outputs.shape)

outputs = model_3(sample)
print(outputs.shape)

outputs = model_4(sample)
print(outputs.shape)

outputs = model_5(sample)
print(outputs.shape)