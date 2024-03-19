import torch
import torch.nn as nn 

# 배치 크기 x 채널 x 높이 x 너비
inputs = torch.Tensor(1, 1, 28, 28)
print('tensor size : {}'.format(inputs.shape))

# 2. 합성곱층과 풀링 선언

# 첫번째 합성곱층
conv1 = nn.Conv2d(1, 32, 3, padding = 1)
print(conv1)

# 두 번째 합성곱층
conv2 = nn.Conv2d(32, 64, kernal_size = 3, padding = 1)
print(conv2)

# 맥스 풀링
# 정수 하나를 인자로 넣으면 커널 사이즈와 스트라이드가 둘 다 해당값으로 지정
pool = nn.MaxPool2d(2)
print(pool)

# 3. 구현체를 연결하여 모델 만들기
out = conv1(inputs)
print(out.shape)

out = pool(out)
print(out.shape)

out = conv2(out)
print(out.shape)

out = pool(out)
print(out.shape)

out.size(0)
cout.size(1)
out.size(2)
out.size(3)

# 첫번째 차원인 배치 차원은 그대로 두고 나머지 펼치기
out = out.view(out.size(0), -1)
print(out.shape)

