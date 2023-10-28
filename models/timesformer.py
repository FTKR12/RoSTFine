import torch.nn as nn
from timesformer.models.vit import TimeSformer

# TimeSformer
class MyTimeSformer(nn.Module):
    def __init__(self, args):
        super(MyTimeSformer, self).__init__() 
        self.model = TimeSformer(
            img_size=args.video_size, 
            num_classes=5, 
            num_frames=args.num_frame,
            attention_type='divided_space_time',
            pretrained_model=args.pretrained_timesformer)
        self.head = nn.Linear(768, 5)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, img):
        out, _, _ = self.model(img)
        out = self.head(out[:,0])
        return self.softmax(out)
    
    def attn_map(self, img):
        _, attn_map_s, _ = self.model(img)
        return attn_map_s
