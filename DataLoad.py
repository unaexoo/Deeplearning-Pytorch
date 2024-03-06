import torch
import torch.nn as nn
import torch.nn.functional as F


# TensorDataset과 DataLoader 임포트
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

x_train  =  torch.FloatTensor([[73,  80,  75], 
                               [93,  88,  93], 
                               [89,  91,  90], 
                               [96,  98,  100],   
                               [73,  66,  70]])  
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train,y_train)

# 데이터 로더 (데이터셋, 미니 배치의 크기)
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)

# 모델과 옵티마이저 설계
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-5 )

# 훈련 진행
nb_epochs = 20

for epoch in range(nb_epochs + 1) : 
    for batch_idx, samples in enumerate(dataloader) : 
        print(batch_idx)
        print(samples)

        x_train, y_train = samples
        
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction,y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
        # epoch, nb_epochs, batch_idx+1, len(dataloader),
        # cost.item()
        # ))

# # 임의의 값
# new_var = torch.FloatTensor([[73,80,75]])

# # 입력한 값에 대해서 예측값 y를 리턴받아서 pred_y에 저장
# pred_y = model(new_var)
# print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y) 