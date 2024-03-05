import torch
import torch.nn.functional as F 
torch.manual_seed(1)

# High Level softmax 비용 함수
z = torch.FloatTensor([1,2,3])

hypothesis = F.softmax(z, dim = 0)
print(hypothesis) # 3개의 원소가 0과 1 사이의 값을 가지는 벡터로 변환

hypothesis.sum()

z = torch.rand(3,5,requires_grad=True)

hypothesis = F.softmax(z, dim=1)
print(hypothesis)

# 임의의 레이블 생성
y = torch.randint(5, (3,)).long()
print(y)

# 원핫 인코딩
y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1,y.unsqueeze(1),1) # scatter 함수는 dim, index, src를 인자로 받아서 dim 차원 기준으로 index 위치의 값을 src로 바꿈

# 수식 4가지
# F.log_softmax(z, dim=1)
# (y_one_hot * - F.log_softmax(z, dim=1)).sum(dim=1).mean()
# F.nll_loss(F.log_softmax(z, dim=1), y)
# F.cross_entropy(z, y)
cost = (y_one_hot * - F.log_softmax(z,dim=1)).sum(dim=1).mean()
print(cost)
