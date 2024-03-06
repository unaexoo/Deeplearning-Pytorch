# 다수의 x로부터 y를 예측하는 다중 로지스틱 회귀

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# print(x_train.shape)
# print(y_train.shape)

W = torch.zeros((2,1), requires_grad=True) # size = 2x1
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000

for epoch in range(nb_epochs + 1):
    # cost 계산
    # e^x torch.exp(x)
    # 가설식
    # hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W)+b)))
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)

print(W)
print(b)

# print(hypothesis)

# 오차
losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
# print(losses)

# 평균
cost = losses.mean()
# print(cost)

# 파이토치에서는 로지스틱 회귀의 비용 함수를 이미 제공
# torch.nn.functional as F -> F.binary_cross_entropy(예측값, 실제값)
F.binary_cross_entropy(hypothesis, y_train)