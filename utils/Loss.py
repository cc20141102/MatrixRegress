
import torch
import torch.nn as nn
"""   
预测损失: mse
"""

class AEDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, y_pred, y):
        return self.mse(y_pred, y)