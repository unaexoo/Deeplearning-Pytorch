import torch
torch.manual_seed(5)
print('랜덤 시드가 5일 때')

for i in range(1,3) : 
    print(torch.rand(1))