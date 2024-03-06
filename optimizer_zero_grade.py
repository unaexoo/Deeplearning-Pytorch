import torch
w = torch.tensor(2.0 , requires_grad=True)

nb_epoch = 20
for epoch in range(nb_epoch+1) :
    z = 2*w
    z.backward()
    print('수식을 w로 미분한 값 : {} '.format(w.grad))