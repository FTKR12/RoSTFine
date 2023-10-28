import torch
import torch.nn as nn

# X3D
class MyX3D(nn.Module):
    def __init__(self, args):
        super(MyX3D, self).__init__()
        # pretrain dataset Kinetics 400
        self.model = torch.hub.load(args.pretrained_dir, 'x3d_m', pretrained=args.pretrained)
        self.model.blocks[5].proj = nn.Linear(in_features=2048, out_features=5, bias=True)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, img):
        out = self.model(img)
        return self.softmax(out)