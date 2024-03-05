import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

# torch.nn.Moudle을 상속받는 파이썬 클래스
class LinearRegressionModel(nn.Module) : 
    def __init__(self) :
        super().__init__()
        self.linear = nn.Linear(1,1) 
        
    def forward(self,x) : 
        return self.linear(x)
    
model = LinearRegressionModel()