import numpy as np
import torch
import torch.nn as nn
from .loss import JSDivLoss
from sklearn.metrics import balanced_accuracy_score

class Metrics(nn.Module):
    def __init__(self, use_metrics):
        super(Metrics, self).__init__()
        self.metrics = use_metrics.split('+')
    
    def forward(self, out, y):
        result = {}
        if 'mse' in self.metrics:
            result['mse'] = self.calc_mse(out, y)
        if 'js' in self.metrics:
            result['js'] = self.calc_js(out, y)
        if 'balanced_acc' in self.metrics:
            result['bacc_1st'] = self.calc_balanced_acc(out, y, 0) 
            result['bacc_2nd'] = self.calc_balanced_acc(out, y, 1) 
            result['bacc_3rd'] = self.calc_balanced_acc(out, y, 2) 
            result['bacc_4th'] = self.calc_balanced_acc(out, y, 3) 
            result['bacc_5th'] = self.calc_balanced_acc(out, y, 4)
            result['bacc_avg'] = np.mean([result['bacc_1st'],result['bacc_2nd'],result['bacc_3rd'],result['bacc_4th'],result['bacc_5th']], axis=0)
        return result
        
    def calc_mse(self, out, y):
        out = torch.tensor(out)
        y = torch.tensor(y)
        func = nn.MSELoss()
        return func(out,y).item()
    
    def calc_js(self, out, y):
        out = torch.tensor(out)
        y = torch.tensor(y)
        func = JSDivLoss()
        return func(out,y).item()
    
    def calc_balanced_acc(self, out, y, i):
        out = torch.tensor(out)
        y = torch.tensor(y)
        _, out_sorted = torch.sort(out, descending=False, dim=1)
        _, y_sorted = torch.sort(y, descending=False, dim=1)
        return balanced_accuracy_score(y_sorted[:,i].tolist(), out_sorted[:,i].tolist())