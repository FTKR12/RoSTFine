import torch
import torch.nn as nn

# 3D Resnet
class MySlowFast(nn.Module):
    def __init__(self, args):
        super(MySlowFast, self).__init__()
        self.model = torch.hub.load(args.pretrained_dir, 'slowfast_r50', pretrained=args.pretrained)
        self.model.blocks[6].proj = nn.Linear(in_features=2304, out_features=5, bias=True)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, img):
        out = self.model(img)
        return self.softmax(out)