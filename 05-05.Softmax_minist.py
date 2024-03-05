import torch
import torchvision.datasets as dsets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn 
import matplotlib.pyplot as plt 
import random

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

# 랜덤 시드 고정
random.seed(777)
torch.manual_seed(777)
if device == 'cuda' :
    torch.cuda.manual_seed_all(777)


# hyperparameters
training_epochs = 15
batch_size = 100

# MINIST dataset
# root : mnist 데이터를 다운로드 받을 경로
# train = True -> 훈련 데이터 리턴, train = False -> 테스트 데이터
# transform : 현재 데이터를 파이토치 텐서로 변환해줌
# download : 해당 경로에 MNIST 데이터가 없다면 다운로드 받겠다
mnist_train = dsets.MNIST(root = 'MNIST_data/', train = True, transform=transforms.ToTensor(),download=True)
mnist_test = dsets.MNIST(root = 'MNIST_data/', train = False, transform=transforms.ToTensor(),download=True)

# dataset loader
# dataset : 로드할 대상
# batch_size : 배치크기
# shuffle : 매 에포크마다 미니 배치를 셔플할 것인지의 여부
# drop_last : 마지막 배치를 버릴 것인지
data_loader = DataLoader(dataset = mnist_train, batch_size= batch_size, shuffle=True, drop_last=True)

# to() : 연산을 어디서 수행할지 정함
linear = nn.Linear(784, 10, bias=True).to(device)

# 비용 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device) # 내부에 소프트맥스 함수 포함
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.1)

for epoch in range(training_epochs) :
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader : 
        # 배치 크기가 100이므로 아래의 연산에서 X는 (100,784)의 텐서가 됨
        X = X.view(-1, 28 * 28).to(device)
        # 레이블은 원핫 인코딩이 된 상태가 아니라 0~9의 정수
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch
    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning finished')

# 테스트 데이터를 사용하여 모델을 테스트
with torch.no_grad() : # torch.no_grad()를 하면 gradient 계산을 수행하지 않음
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction,1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: ', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아 예측
    r = random.randint(0,len(mnist_test)-1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label : ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction : ', torch.argmax(single_prediction,1).item())

    plt.imshow(mnist_test.test_data[r:r+1].view(28,28), cmap='Grays', interpolation='nearest')
    plt.show()