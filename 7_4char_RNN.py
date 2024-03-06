import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 훈련 데이터 전처리
# 입력 데이터와 레이블 데이터에 대해 문자 집합(voabulary) 생성
# 여기선 문자 집합은 중복을 제거한 문자들의 집합
input_str = 'apple'
label_str = 'pple!'
char_vocab = sorted(list(set(input_str+label_str)))
vocab_size = len(char_vocab)
print('문자 집합의 크기 : {}'.format(vocab_size))

# 입력 : 원핫벡터 -> 입력의 크기는 문자 집합의 크기
input_size = vocab_size
hidden_size = 5
output_size = 5
learning_rate = 0.1

# 문자 집합에 고유 정수 부여
char_to_index = dict((c,i) for i,c in enumerate(char_vocab))
print(char_to_index)

# 예측 결과(를 다시 문자 시퀀스로 보기 위한 정수 -> 문자
index_to_char={}
for key, value in char_to_index.items():
    index_to_char[value] = key
print(index_to_char)

# 입력 데이터와 레이블 데이터의 각 문자들 정수 맵핑
x_data = [char_to_index[c] for c in input_str]
y_data = [char_to_index[c] for c in label_str]
print(x_data)
print(y_data)

x_data = [x_data]
y_data = [y_data]
print(x_data)
print(y_data)
# 입력 시퀀의 각 문자들을 원 핫 벡터로 변경
x_one_hot = [np.eye(vocab_size)[x] for x in x_data]
print(x_one_hot)

# 입력 데이터와 레이블 데이터를 텐서로 변경
X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
print('훈련 데이터의 크기 : {}'.format(X.shape))
print('레이블의 크기 : {}'.format(Y.shape))

# 2. 모델 구현
# fc : 완전 연결층(fully-connected layer) -> 출력층으로 사용

class Net(torch.nn.Module):
    def __init__(self,input_size,hidden_size, output_size):
        super(Net,self).__init__()
        
        # RNN 셀 구현
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first = True)
        
        # 출력층 구현
        self.fc = torch.nn.Linear(hidden_size, output_size, bias = True)
        
    # 구현한 RNN 셀과 출력층을 연결
    def forward(self, x) :
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

    
net = Net(input_size, hidden_size, output_size)
outputs = net(X)
print(outputs.shape) 
# torch.size([1,5,5]) - >[배치 차원, 시점(timesteps), 출력의 크기]

print(outputs.view(-1, input_size).shape) # 2차원 텐서로 변환
print(Y.shape)
print(Y.view(-1).shape)

# 옵티마이저와 손실 함수
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

for i in range(100) : 
    optimizer.zero_grad()
    outputs = net(X)
    
    # view를 하는 이유는 Batch 차원 제거를 위해
    loss = criterion(outputs.view(-1, input_size), Y.view(-1))
    loss.backward() # 기울기 계산
    optimizer.step() # 파라미터 업데이트
    
    # 모델 예측 확인용
    # 최종 예측값인 time - step 별 5차원 벡터에 대해 가장 높은 값 인덱스 선택
    result = outputs.data.numpy().argmax(axis = 2)
    result_str = ''.join([index_to_char[c] for c in np.squeeze(result)])
    print(i, "loss: ", loss.item(), "prediction: ", result, "true Y: ", y_data, "prediction str: ", result_str)