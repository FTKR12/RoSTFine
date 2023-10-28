import torch
import torch.nn as nn
import einops
from einops import rearrange
from timesformer.models.vit import TimeSformer

# Original Model
class RoSTFine(nn.Module):
    def __init__(self, args):
        super(RoSTFine, self).__init__()
        self.args = args
        self.model = TimeSformer(
            img_size=args.video_size, 
            num_classes=5, 
            num_frames=args.num_frame,
            attention_type='divided_space_time',
            pretrained_model=args.pretrained_timesformer)
        self.token_select = TokenSelection(args)

        self.norm_s1 = nn.LayerNorm(self.model.model.embed_dim)
        #self.mhsa_s = MultiheadAttention(self.model.model.embed_dim, args.num_heads)
        self.mhsa_s = nn.ModuleList([MultiheadAttention(self.model.model.embed_dim, args.num_heads) for _ in range(args.num_blk)])
        self.norm_s2 = nn.LayerNorm(self.model.model.embed_dim)
        self.mlp_s = Mlp(self.model.model.embed_dim)

        self.norm_t1 = nn.LayerNorm(self.model.model.embed_dim)
        #self.mhsa_t = MultiheadAttention(self.model.model.embed_dim, args.num_heads)
        self.mhsa_t = nn.ModuleList([MultiheadAttention(self.model.model.embed_dim, args.num_heads) for _ in range(args.num_blk)])
        self.norm_t2 = nn.LayerNorm(self.model.model.embed_dim)
        self.mlp_t = Mlp(self.model.model.embed_dim)

        self.head_g = nn.Linear(768, 5)
        self.head_s = nn.Linear(768, 5)
        self.head_t = nn.Linear(768, 5)
        self.softmax = nn.Softmax(dim=1)
        #self.token_select_t = TimeTokenSelection(args)       
        #self.mlp2 = Mlp()
        #self.softmax = nn.Softmax(dim=1)
    
    def forward(self, img):
        B = img.size(0)

        out, attn_map_s, _ = self.model(img)
        feature_global = out[:,0]
        #print(featrue_global.size()) # (B, 768)
        #print('attn_map_s',attn_map_s.size()) # torch.Size([B*num_frame, heads(12), 197, 197])
        #print('attn_map_t',attn_map_t.size()) # torch.Size([B, 196, heads(12), frames, frames])

        space_selected, _ = self.token_select(out[:,1:], attn_map_s) # (8,8*topk1,768)

        # time attn
        #out_attn_t = self.mhsa_t(self.norm_t1(torch.cat((feature_global.unsqueeze(1), space_selected), dim=1)))
        x = torch.cat((feature_global.unsqueeze(1), space_selected), dim=1)
        for blk in self.mhsa_t:
            x, _ = blk(self.norm_t1(x))
        out_attn_t = self.mlp_t(self.norm_t2(x))
        time_global = out_attn_t[:,0]

        # space attn
        cls_token = feature_global.repeat(self.args.num_frame, 1, 1).transpose(1,0)
        cls_token = rearrange(cls_token, 'b t d -> (b t) d', b=B, t=self.args.num_frame, d=768).unsqueeze(1)
        space_selected = rearrange(space_selected, 'b (t topk1) d -> (b t) topk1 d', b=B, t=self.args.num_frame, topk1=self.args.topk, d=768)
        #out_attn_s = self.mhsa_s(self.norm_s1(torch.cat((cls_token, space_selected), dim=1)))
        x = torch.cat((cls_token, space_selected), dim=1)
        for blk in self.mhsa_s:
            x, _ = blk(self.norm_s1(x))
        out_attn_s = self.mlp_s(self.norm_s2(x))
        space_global = rearrange(out_attn_s[:,0], '(b t) d -> b t d', b=B, t=self.args.num_frame, d=768)
        space_global = torch.mean(space_global,1)

        out_g = self.softmax(self.head_g(feature_global))
        out_s = self.softmax(self.head_s(space_global))
        out_t = self.softmax(self.head_t(time_global))
        return out_g, out_s, out_t, feature_global, space_global, time_global

    def attn_map(self, x):
        B = x.size(0)
        out, attn_map_s, _ = self.model(x)
        feature_global = out[:,0]

        space_selected, indices = self.token_select(out[:,1:], attn_map_s) # (8,8*topk1,768)

        # space attn
        cls_token = feature_global.repeat(self.args.num_frame, 1, 1).transpose(1,0)
        cls_token = rearrange(cls_token, 'b t d -> (b t) d', b=B, t=self.args.num_frame, d=768).unsqueeze(1)
        space_selected = rearrange(space_selected, 'b (t topk1) d -> (b t) topk1 d', b=B, t=self.args.num_frame, topk1=self.args.topk, d=768)
        #out_attn_s = self.mhsa_s(self.norm_s1(torch.cat((cls_token, space_selected), dim=1)))
        x = torch.cat((cls_token, space_selected), dim=1)
        for blk in self.mhsa_s:
            _, attn_map = blk(self.norm_s1(x))
        attn_map = rearrange(attn_map[:,:,0,1:], '(b t) h m -> b t h m', b=B)
        attn_map = torch.sum(attn_map, dim=2)

        attn_map_zero = torch.zeros(B,self.args.num_frame,196).to(self.args.device)
        for i, batch_idx in enumerate(indices):
            for frame in range(self.args.num_frame):
                attn_map_zero[i,frame,batch_idx[frame]] = attn_map[i,frame]
        return attn_map_zero

# Space Token Selection Modle
class TokenSelection(nn.Module):
    def __init__(self, args):
        super(TokenSelection, self).__init__()
        self.args = args
    
    def forward(self, tokens, attn_maps):
        B = tokens.size(0)
        tokens = rearrange(tokens, 'b (t p) d -> b t p d', b=B,t=self.args.num_frame,p=196,d=768)
        #attn_maps = rearrange(attn_maps, 'b t l h n m -> (b t) l h n m')
        attn_maps = torch.sum(attn_maps[:,:,self.args.top_attn:,:,0,1:], dim=3)
        attn_maps = torch.sum(attn_maps, dim=2)
        indices = torch.topk(attn_maps, self.args.topk, dim=2, sorted=True).indices

        out = []
        for i in range(B):
            out.append(vector_gather(tokens[i],indices[i]).unsqueeze(0))
        out = rearrange(torch.cat(out, dim=0), 'b t topk1 d -> b (t topk1) d',b=B,t=self.args.num_frame,topk1=self.args.topk,d=768)
        #out = self.mhsa(out)
        return out, indices

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_map)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x, attn_map

def vector_gather(vectors, indices):
    """
    Gathers (batched) vectors according to indices.
    Arguments:
        vectors: Tensor[N, L, D]
        indices: Tensor[N, K] or Tensor[N]
    Returns:
        Tensor[N, K, D] or Tensor[N, D]
    """
    N, L, D = vectors.shape
    squeeze = False
    if indices.ndim == 1:
        squeeze = True
        indices = indices.unsqueeze(-1)
    N2, K = indices.shape
    assert N == N2
    indices = einops.repeat(indices, "N K -> N K D", D=D)
    out = torch.gather(vectors, dim=1, index=indices)
    if squeeze:
        out = out.squeeze(1)
    return out