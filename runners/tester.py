import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from gradcam.utils import visualize_cam
from .metrics import Metrics
from prettytable import PrettyTable
import logging
log = logging.getLogger(__name__)

class Tester():
    def __init__(self, args, fold, valid_dataloader, model):

        self.device = args.device
        self.output_dir = args.output_dir
        self.num_frame = args.num_frame
        self.load_dir = args.load_dir
        self.valid_dataloader = valid_dataloader
        self.model_name = args.model_name
        self.model = model
        self.top_attn = args.top_attn
        self.fold = fold
        self.metrics = Metrics(args.use_metrics)
        self.plot_w = args.plot_w
        self.plot_h = args.plot_h

    def run(self):
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(f'{self.load_dir}/checkpoint/model_{self.fold}.bin'))
        self.model.eval()
        
        results = {'id': [], 'out': [], 'y': []}
        with torch.no_grad():
            for x, _, y, id in tqdm(self.valid_dataloader):
                if self.model_name == 'slowfast': x = [k.to(self.device) for k in x]
                else: x = x.to(self.device)
                y = y.to(self.device)
                out, attn_map = self.predict(x,y)
                results['id'] += list(id)
                results['out'] += out.tolist()
                results['y'] += y.tolist()
                
                if self.model_name in ['timesformer', 'rostfine']:
                    self.plot_attn(id, x, attn_map)
                
        evaluated = self.metrics(results['out'], results['y'])
        self.plot(results)
        return evaluated

    def predict(self, x, y):
        if self.model_name == 'slowfast':
            x = [i for i in x]
        elif self.model_name == 'vivit':
            x = x.permute(0,2,1,3,4)
        elif self.model_name =='vgg':
            x = x[:,:,0]

        if self.model_name == 'rostfine':
            out_g, out_s, out_t, feature_global, space_global, time_global = self.model(x)
            out = (out_g + out_s + out_t) / 3
            attn_map = self.model.attn_map(x)
        elif self.model_name == 'timesformer':  
            out = self.model(x)
            attn_map = self.model.attn_map(x)
        else:
            out = self.model(x)
            attn_map = None
            
        return out, attn_map

    def plot(self, result):
        for id, true, pred in zip(result['id'], result['y'], result['out']):
            plt.figure(figsize=(self.plot_w, self.plot_h))
            plt.suptitle(f"ID: {id}")

            plt.subplot(1,2,1)
            plt.title("true")
            plt.yticks(color="None")
            plt.tick_params(length=0)
            plt.ylim(0,1.0)
            plt.bar(['A','B','C','D','E'], true)

            plt.subplot(1,2,2)
            plt.title("pred")
            plt.yticks(color="None")
            plt.ylim(0,1.0)
            plt.tick_params(length=0)
            plt.bar(['A','B','C','D','E'], pred)

            plt.savefig(f'{self.load_dir}/samples/{id}.png')
            plt.close()

    def plot_attn(self, ids, xs, attn_maps):
        if self.model_name == 'timesformer':
            attn_maps = torch.sum(attn_maps[:,:,self.top_attn:,:,0,1:], dim=2)
            attn_maps = torch.sum(attn_maps, dim=2)
        
        attn_maps = attn_maps.detach().cpu().numpy()
        for id, x, attn_map in zip(ids, xs, attn_maps):
            for frame in range(self.num_frame):
                attn_map_ = attn_map[frame].reshape(14,14)
                attn_map_ = cv2.resize(attn_map_ / attn_map_.max(), (x.size(2), x.size(3)))[..., np.newaxis]
                attn_map_ = torch.from_numpy(attn_map_.astype(np.float32))
                _, result = visualize_cam(attn_map_, 0.5*x[:,frame]+0.8)

                plt.figure(figsize=(8,8))
                plt.title(f"{id}")
                plt.xticks(color="None")
                plt.yticks(color="None")
                plt.tick_params(length=0)

                plt.imshow(result.numpy().transpose(1, 2, 0))

                plt.savefig(f'{self.load_dir}/attn/{id}_{frame}.png')
                plt.close()