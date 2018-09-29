import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn.functional as nn
import torch.utils.data as Data

import pandas as pd
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")


# ===================Train Data======================= #
#
#  输入10个采样点，预测5个采样点。由于每个
#  采样点都包含3个坐标(x,y,z)，所以需要输入
#  维度为30，输出维度为15。
#
#  每段轨迹长度3秒，共90个采样点。最后10个
#  采样点不需要作为输入，前10个采样点不需要
#  作为预测的真实值输出。前80个采样点作为输
#  入
#  
#  N = landmarks.shape[0] - landmarks.shape[0]/9
#
#  range取值是轨迹的条数
# =====================================================

# FC-net architecture
D_in, H_in, H_out, D_out = 30, 15, 15, 15 
Z_dim, Learning_rate = 3, 1e-3
#D_in, H_in, H_out, D_out = 20, 15, 15, 10

landmarks_frame = pd.read_csv('Try.csv')
landmarks = landmarks_frame.as_matrix().astype('float')

# Add agument data
plus = np.random.randint(low=3, high=4, size=landmarks.shape)
plus = plus/10000
plus = landmarks + plus
tempo = np.append(landmarks, plus, axis=0)

sub = np.random.randint(low=1, high=2, size=landmarks.shape)
sub = sub/10000
sub = landmarks - sub
landmarks = np.append(tempo, sub, axis=0)
print(landmarks.shape)

# Normalization
#maxdata = np.array([1.5, 0.4, 1.2])
#mindata = np.array([0.8, -0.5, 0.6])
maxdata = np.max(landmarks, axis=0)
mindata = np.min(landmarks, axis=0)
landmarks = (landmarks - mindata)/(maxdata - mindata)
landmarks = landmarks.astype('float').reshape(-1, D_in)

N = landmarks.shape[0] * 8 / 9
N = int(N)
count = landmarks.shape[0] / 9
count = int(count)
#print('Landmarks shape: {}'.format(landmarks.shape))
x = torch.zeros(N, D_in)
y = torch.zeros(N, D_out)

for i in range(count):
    x[i*8:i*8+8] = Variable(torch.from_numpy(landmarks[i*9:i*9+8]), requires_grad = False)
    y[i*8:i*8+8] = Variable(torch.from_numpy(landmarks[i*9+1:i*9+9, :D_out]), requires_grad = False)
print(x.shape)

torch.manual_seed(1)    # reproducible
torch_dataset = Data.TensorDataset(x, y)
# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=8,               # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)


# ===================Validation Data======================= #
landmarks_frame_val = pd.read_csv('Val.csv')
landmarks_val = landmarks_frame_val.as_matrix().astype('float')

landmarks_val = (landmarks_val - mindata)/(maxdata - mindata)
landmarks_val = landmarks_val.astype('float').reshape(-1, D_in)

#print('Landmarks shape: {}'.format(landmarks_val.shape))

N_val = landmarks_val.shape[0] * 8 / 9
N_val = int(N_val)
count_val = landmarks_val.shape[0] / 9
count_val = int(count_val)

x_val = torch.zeros(N_val, D_in)
y_val = torch.zeros(N_val, D_out)

for i in range(count_val):
    x_val[i*8:i*8+8] = Variable(torch.from_numpy(landmarks_val[i*9:i*9+8]), requires_grad = False)
    y_val[i*8:i*8+8] = Variable(torch.from_numpy(landmarks_val[i*9+1:i*9+9, :D_out]), requires_grad = False)

print(x_val.shape)