import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

# 현재 실습하고 있는 파이썬 코드를 재실행해도 다음에도 같은 결과가 나오도록 랜덤 시드(random seed)를 줌
torch.manual_seed(1)

# 1. 변수 선언

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[2],[4],[6]])

# x_train과 x_train의 크기(shape)
print(x_train)
print(x_train.shape)

# y_train과 y_train 크기(shape)
print(y_train)
print(y_train.shape)

# 2. 가중치와 편향의 초기화
# 가중치 W = 0으로 초기화
W = torch.zeros(1,requires_grad=True)

# 가중치 W 출력
print(W)

# 편향 b = 0 으로 초기화 -> 학습을 통해 값이 변경되는 변수임을 명시함
b = torch.zeros(1, requires_grad= True)
print(b)

# 3. 직선의 방정식에 해당되는 가설을 선언
hypothesis = x_train * W + b
print(hypothesis)

# 4. 비용 함수 선언
# torch.mean으로 평균 구함
cost = torch.mean((hypothesis - y_train)**2)
print(cost)

# 5. 경사 하강법 구현하기
optimizer = optim.SGD([W,b], lr = 0.01)

# gradient를 0으로 초기화
optimizer.zero_grad()

# 비용 함수를 미분하여 gradient 계산
cost.backward()

# W와 b를 업데이트
optimizer.step()