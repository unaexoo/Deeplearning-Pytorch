import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np

# 1. 훈련 데이터 전처리
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

# 문자 집합 생성, 고유 정수 부여
# 중복 제거 문자 집합 
char_set = list(set(sentence)) 

# 각 문자에 정수 인코딩
char_dic = {c:i for i,c in enumerate(char_set)}
print(char_dic) # 공백도 원소

# 매 시점마다 들어갈 입력의 크기
dic_size = len(char_dic)
print('문자 집합의 크기 : {}'.format(dic_size))

# 하이퍼파라미터 설정
hidden_size = dic_size
sequence_length = 10
learning_rate = 0.1

# 샘플들을 잘라서 데이터를 만듦
# 데이터 구성
x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x_data.append([char_dic[c] for c in x_str])  # x str to index
    y_data.append([char_dic[c] for c in y_str])  # y str to index
  
print(x_data[0])
print(y_data[0])

# 입력 시퀀스에 대해 원핫 인코딩을 수행하고 입력 데이터와 레이블 데이터를 텐서로 변환
x_one_hot = [np.eye(dic_size)[x] for x in x_data] # x 데이터는 원-핫 인코딩
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)


print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))
print(X[0])
print(Y[0])


# 2. 모델 구현
# -> 단위 RNN과 동일하지만 은닉층을 두 개 쌓음
class Net(torch.nn.Module) : 
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net,self).__init__()
        
        # num_layers는 은닉층을 몇 개 쌓을 것인지 의미
        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers = layers, batch_first = True)
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim, bias = True)
        
    def forward(self,x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x
    
net = Net(dic_size,hidden_size,2)

# 비용함수와 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

outputs = net(X)
print(outputs.shape)

print(outputs.view(-1, dic_size).shape) # 2차원 텐서로 변환.
print(Y.shape)
print(Y.view(-1).shape)


for i in range(100) :
    optimizer.zero_grad()
    outputs = net(X)
    loss = criterion(outputs.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()
    
    results = outputs.argmax(dim=2)
    predict_str = ""
    for j, result in enumerate(results) : 
        # 처음엔 예측 결과를 전부 가져오고, 이후에는 마지막 글자만 반복 추가
        if j==0 : 
            predict_str += ''.join([char_set[t] for t in result])
        else :
            predict_str += char_set[result[-1]]

print(predict_str)