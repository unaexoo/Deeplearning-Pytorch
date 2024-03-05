import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim  

torch.manual_seed(1)

# 훈련 데이터 선언
x_train = torch.FloatTensor([[73,  80,  75], 
                             [93,  88,  93], 
                             [89,  91,  80], 
                             [96,  98,  100],   
                             [73,  66,  70]])  
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

# 가중치 w 와 편향 b 초기화
W = torch.zeros((3,1),requires_grad=True)
b = torch.zeros(1,requires_grad = True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr = 1e-5)

nb_epochs = 20

for epoch in range(nb_epochs + 1):
    # H(x) 계산. 편향 b는 브로드 캐스팅 되어 각 샘플에 더해짐
    # 행렬곱으로 가설 선언  
    hypothesis = x_train.matmul(W) + b
    
    # cost 계산
    cost = torch.mean((hypothesis - y_train)**2)
    
    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))

