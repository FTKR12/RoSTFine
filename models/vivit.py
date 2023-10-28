from torch import nn
import sys
sys.path.append('ViViT')
from video_transformer import ViViT

# ViViT
class MyViViT(nn.Module):
    def __init__(self, args):
        super(MyViViT, self).__init__()
        # pretrain dataset kinetics 400
        self.model = ViViT(img_size=args.video_size, num_frames=16, attention_type='fact_encoder', pretrain_pth=args.pretrained_vivit)
        self.linear = nn.Linear(768, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, img):
        out = self.model(img)
        out = self.linear(out)
        return self.softmax(out)