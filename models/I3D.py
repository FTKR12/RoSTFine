import torch
import torch.nn as nn


# I3D
class MyI3D(nn.Module):
    def __init__(self, args):
        super(MyI3D, self).__init__()
        # pretrain 
        self.model = torch.hub.load(args.pretrained_dir, 'i3d_r50', pretrained=args.pretrained)
        self.model.blocks[6].proj = nn.Linear(in_features=2048, out_features=5, bias=True)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, img):
        out = self.model(img)
        return self.softmax(out)