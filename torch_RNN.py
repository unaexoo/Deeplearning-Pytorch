import torch
import torch.nn as nn

# 입력 크기
input_size = 5

# 은닉 상태의 크기
hidden_size = 8 

# 입력 텐서 : 배치 크기 x 시점의 수 x 매 시점맏 들어가는 입력
inputs = torch.Tensor(1,10,5)

# nn.RNN(입력의 크기, 은닉 상태의 크기, batch_first = True(입력 텐서의 첫번째 차원이 배치 크기임을 알려줌))
cell = nn.RNN(input_size, hidden_size,batch_first=True)

# RNN 셀이 리턴하는 것
# 1. 모든 시점(timesteps)의 은닉 상태들
# 2. 마지막 시점(timestep)의 은닉 상태

# 첫 번째 리턴값
outputs, _status = cell(inputs)
print(outputs.shape)

# 두 번째 리턴값
print(_status.shape)