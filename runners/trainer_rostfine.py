import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import logging
from tensorboardX import SummaryWriter

from .optimizer import build_optim
from .loss import build_loss

logger = logging.getLogger("Sperm-Assessment")

class RoSTFineTrainer():
    def __init__(self, args, fold, train_dataloader, valid_dataloader, model):

        self.device = args.device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model_name = args.model_name
        self.model = model
        self.use_feat = args.use_feat.split('+')
        self.use_div = args.use_div.split('+')
        self.epochs = args.epochs
        self.optimizer = build_optim(args.optim, args.lr, args.momentum, args.weight_decay, self.model)
        self.loss_func = build_loss(args.loss)
        self.div = nn.CosineSimilarity(dim=0)
        self.alpha = args.alpha
        self.fold = fold
        self.train_loss_log = {'vg': [], 'vs': [], 'vt': [], 'all': [], 'avg': [], 'loss+div': []}
        self.valid_loss_log = {'vg': [], 'vs': [], 'vt': [], 'all': [], 'avg': [], 'loss+div': []}
        self.train_div_log = {'gs': [], 'gt': [], 'st': [], 'avg': []}
        self.valid_div_log = {'gs': [], 'gt': [], 'st': [], 'avg': []}

        self.output_dir = args.output_dir
        self.tensorboard_dir = args.tensorboard_dir

    def run(self):
        self.model.to(self.device)
        writer = SummaryWriter(log_dir=f"{self.output_dir}/{self.tensorboard_dir}/fold{self.fold}")
        
        for epoch in tqdm(range(self.epochs)):
            #valid
            self.model.eval()
            with torch.no_grad():
                valid_losses = {'vg': 0, 'vs': 0, 'vt': 0, 'all': 0, 'avg': 0, 'loss+div': 0}
                valid_divs = {'gs': 0, 'gt': 0, 'st': 0, 'avg': 0}
                self.model.eval()
                for x, _, y, _ in self.valid_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    losses, outs, divs = self.predict(x,y)
                    loss = losses['avg'] + self.alpha*divs['avg']
                    
                    valid_losses['vg'] += losses['vg'].item()
                    valid_losses['vs'] += losses['vs'].item()
                    valid_losses['vt'] += losses['vt'].item()
                    valid_losses['all'] += losses['all'].item()
                    valid_losses['avg'] += losses['avg'].item()
                    valid_losses['loss+div'] += loss.item()
                    
                    valid_divs['gs'] += divs['gs'].item()
                    valid_divs['gt'] += divs['gt'].item()
                    valid_divs['st'] += divs['st'].item()
                    valid_divs['avg'] += divs['avg'].item()

            L = self.valid_dataloader.__len__()
            valid_losses['vg'] /=L
            valid_losses['vs'] /=L
            valid_losses['vt'] /=L
            valid_losses['all'] /=L
            valid_losses['avg'] /=L
            valid_losses['loss+div'] /=L
            
            valid_divs['gs'] /=L
            valid_divs['gt'] /=L
            valid_divs['st'] /=L
            valid_divs['avg'] /=L

            self.valid_loss_log['vg'].append(valid_losses['vg'])
            self.valid_loss_log['vs'].append(valid_losses['vs'])
            self.valid_loss_log['vt'].append(valid_losses['vt'])
            self.valid_loss_log['all'].append(valid_losses['all'])
            self.valid_loss_log['avg'].append(valid_losses['avg'])
            self.valid_loss_log['loss+div'].append(valid_losses['loss+div'])

            self.valid_div_log['gs'].append(valid_divs['gs'])
            self.valid_div_log['gt'].append(valid_divs['gt'])
            self.valid_div_log['st'].append(valid_divs['st'])
            self.valid_div_log['avg'].append(valid_divs['avg'])

            self.save_checkpoint(self.valid_loss_log['all'], epoch)

            # train
            train_losses = {'vg': 0, 'vs': 0, 'vt': 0, 'all': 0, 'avg': 0, 'loss+div': 0}
            train_divs = {'gs': 0, 'gt': 0, 'st': 0, 'avg': 0}
            self.model.train()
            for x, _, y, _ in self.train_dataloader:
                y += torch.tensor([0.001,0.001,0.001,0.001,0.001])
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                losses, outs, divs = self.predict(x,y)
                loss = losses['avg'] + self.alpha*divs['avg']
                loss.backward()
                self.optimizer.step()
                
                train_losses['vg'] += losses['vg'].item()
                train_losses['vs'] += losses['vs'].item()
                train_losses['vt'] += losses['vt'].item()
                train_losses['all'] += losses['all'].item()
                train_losses['avg'] += losses['avg'].item()
                train_losses['loss+div'] += loss.item()
                
                train_divs['gs'] += divs['gs'].item()
                train_divs['gt'] += divs['gt'].item()
                train_divs['st'] += divs['st'].item()
                train_divs['avg'] += divs['avg'].item()

            L = self.train_dataloader.__len__()
            train_losses['vg'] /=L
            train_losses['vs'] /=L
            train_losses['vt'] /=L
            train_losses['all'] /=L
            train_losses['avg'] /=L
            train_losses['loss+div'] /=L
            
            train_divs['gs'] /=L
            train_divs['gt'] /=L
            train_divs['st'] /=L
            train_divs['avg'] /=L

            self.train_loss_log['vg'].append(train_losses['vg'])
            self.train_loss_log['vs'].append(train_losses['vs'])
            self.train_loss_log['vt'].append(train_losses['vt'])
            self.train_loss_log['all'].append(train_losses['all'])
            self.train_loss_log['avg'].append(train_losses['avg'])
            self.train_loss_log['loss+div'].append(train_losses['loss+div'])

            self.train_div_log['gs'].append(train_divs['gs'])
            self.train_div_log['gt'].append(train_divs['gt'])
            self.train_div_log['st'].append(train_divs['st'])
            self.train_div_log['avg'].append(train_divs['avg'])
            
            logger.info(f"[EPOCH]{epoch+1:3}    (train_loss){train_losses['all']: 10.8f}    (val_loss){valid_losses['all']: 10.8f}")
            writer.add_scalars(f"Loss{self.fold}", {"train_loss+div": train_losses['loss+div'], "val_loss+div": valid_losses['loss+div'], \
                "train_all": train_losses['all'], "valid_all": valid_losses['all'], \
                    "train_avg": train_losses['avg'], "train_vg": train_losses['vg'], "train_vs": train_losses['vs'], "train_vt": train_losses['vt'], \
                        "valid_avg": valid_losses['avg'], "valid_vg": valid_losses['vg'], "valid_vs": valid_losses['vs'], "valid_vt": valid_losses['vt']}, epoch)
            writer.add_scalars(f"Div{self.fold}", \
                {"train_gs": train_divs['gs'], "train_gt": train_divs['gt'], "train_st": train_divs['st'], \
                    "valid_gs": valid_divs['gs'], "valid_gt": valid_divs['gt'], "valid_st": valid_divs['st']}, epoch)

        logger.info(f'(min_loss){self.min_loss: 10.8f}')
        #plot
        for k in self.train_loss_log.keys():
            self.plot_cv(self.train_loss_log[k], self.valid_loss_log[k], f"loss_{k}")
        for k in self.train_div_log.keys():
            self.plot_cv(self.train_div_log[k], self.valid_div_log[k], f"div_{k}")
        # save loss log
        loss_log = {'train loss': self.train_loss_log, 'valid loss': self.valid_loss_log}
        with open(f'{self.output_dir}/loss/log_loss_{str(self.fold)}.pkl', 'wb') as f:
            pickle.dump(loss_log, f)
        div_log = {'train div': self.train_div_log, 'valid div': self.valid_div_log}
        with open(f'{self.output_dir}/loss/log_div_{str(self.fold)}.pkl', 'wb') as f:
            pickle.dump(div_log, f)

    def predict(self, x, y):
        
        out_g, out_s, out_t, vg, vs, vt = self.model(x)
        
        out = None
        outs = {'vg': out_g, 'vs': out_s, 'vt': out_t} 
        for k in self.use_feat:
            if out == None:
                out = outs[k]
            else:
                out = out + outs[k]
        out = out/len(self.use_feat)
        outs['out'] = out
        
        loss = None
        losses = {'vg': self.loss_func(y, out_g), 'vs': self.loss_func(y, out_s), 'vt': self.loss_func(y, out_t), 'all': self.loss_func(y, out)}
        for k in self.use_feat:
            if loss == None:
                loss = losses[k]
            else:
                loss = loss + losses[k]
        loss = loss/len(self.use_feat)
        losses['avg'] = loss
        
        div = None
        B = vg.size(0)
        divs = {
            'gs': sum([torch.abs(self.div(vg[i],vs[i])) for i in range(B)])/B,
            'gt': sum([torch.abs(self.div(vg[i],vt[i])) for i in range(B)])/B,
            'st': sum([torch.abs(self.div(vs[i],vt[i])) for i in range(B)])/B
            }
        for k in self.use_div:
            if div == None:
                div = divs[k]
            else:
                div += divs[k]
        div /= len(self.use_div)
        divs['avg'] = div
            
        return losses, outs, divs

    def save_checkpoint(self, valid_loss_log, epoch):
        if epoch == 0:
            torch.save(self.model.state_dict(), f"{self.output_dir}/checkpoint/model_{str(self.fold)}.bin")
            self.min_loss = valid_loss_log[epoch]
        elif valid_loss_log[epoch] < self.min_loss:
            torch.save(self.model.state_dict(), f"{self.output_dir}/checkpoint/model_{str(self.fold)}.bin")
            self.min_loss = valid_loss_log[epoch]
        elif valid_loss_log[epoch] > self.min_loss:
            pass
        return

    def plot_cv(self, train_loss_log, valid_loss_log, name):
        plt.figure(figsize=(6.4,4.8))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(train_loss_log, label='train loss')
        plt.plot(valid_loss_log, label='test loss')
        plt.legend()
        plt.savefig(f"{self.output_dir}/learning_curves/learning_curve{str(self.fold)}_{name}.png")
        plt.close()
    