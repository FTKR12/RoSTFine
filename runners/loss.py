import torch
import torch.nn as nn

def build_loss(loss_name):
    # TODO: add loss func
    loss_factory = {
        'mse': nn.MSELoss(),
        'js': JSDivLoss()
    }
    return loss_factory[loss_name]

class JSDivLoss(nn.Module):
    def __init__(self) -> None:
        super(JSDivLoss, self).__init__()
        
    def forward(self,p,q) -> torch.Tensor:
        return self.js_div(p,q)
    
    def kl_div(self,p,q) -> torch.Tensor:
        kl = p*torch.log((p+0.00001)/q)
        kl = torch.sum(torch.sum(kl, dim=1), dim=0)/p.size(0)
        return kl
    
    def js_div(self,p,q) -> torch.Tensor:
        return self.kl_div(p,(p+q)/2) + self.kl_div(q,(p+q)/2)/2
    