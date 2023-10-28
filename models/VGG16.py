import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet18, resnet50

# vgg16
class MyVgg(nn.Module):
    def __init__(self, args):
        super(MyVgg, self).__init__()        
        self.features = vgg16(pretrained=args.pretrained).features
        self.avgpool = vgg16(pretrained=args.pretrained).avgpool
        self.flatten_map = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=512*7*7, out_features=512, bias=True),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=5, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, img):
        img = self.features(img)
        img = self.avgpool(img)
        img = self.flatten_map(img)
        out = self.classifier(img)
        return out