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

class Trainer():
    def __init__(self, args, fold, train_dataloader, valid_dataloader, model):
        
        self.device = args.device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model_name = args.model_name
        self.model = model
        self.epochs = args.epochs
        self.optimizer = build_optim(args.optim, args.lr, args.momentum, args.weight_decay, self.model)
        self.loss_func = build_loss(args.loss)
        self.fold = fold
        self.train_loss_log = []
        self.valid_loss_log = []
        
        self.output_dir = args.output_dir
        self.tensorboard_dir = args.tensorboard_dir


    def run(self):
        self.model.to(self.device)
        writer = SummaryWriter(log_dir=f"{self.output_dir}/{self.tensorboard_dir}/fold{self.fold}")
        
        for epoch in tqdm(range(self.epochs)):
            #valid
            self.model.eval()
            with torch.no_grad():
                valid_loss = 0
                for x, _, y, _ in self.valid_dataloader:
                    if self.model_name == 'slowfast': x = [k.to(self.device) for k in x]
                    else: x = x.to(self.device)
                    y = y.to(self.device)
                    loss, _ = self.predict(x,y)
                    valid_loss += loss.item()
                valid_loss /= self.valid_dataloader.__len__()
            self.valid_loss_log.append(valid_loss)
            self.save_checkpoint(self.valid_loss_log, epoch)

            # train
            train_loss = 0
            self.model.train()
            for x, _, y, _ in self.train_dataloader:
                if self.model_name == 'slowfast': x = [k.to(self.device) for k in x]
                else: x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                loss, _ = self.predict(x,y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= self.train_dataloader.__len__()
            self.train_loss_log.append(train_loss)

            logger.info(f"[EPOCH]{epoch+1:3}    (train_loss){train_loss: 10.8f}    (val_loss){valid_loss: 10.8f}")
            writer.add_scalars(f"Loss{self.fold}", {"train":train_loss, "val":valid_loss}, epoch)
        
        logger.info(f'(min_loss){self.min_loss: 10.8f}')
        #plot
        self.plot_cv(self.train_loss_log, self.valid_loss_log)
        # save loss log
        loss_log = {'train loss': self.train_loss_log, 'valid loss': self.valid_loss_log}
        with open(f'{self.output_dir}/loss/log_loss_{str(self.fold)}.pkl', 'wb') as f:
            pickle.dump(loss_log, f)
        writer.export_scalars_to_json(f"{self.output_dir}/{self.tensorboard_dir}/fold{self.fold}/all_scalars.json")
        writer.close()

    def predict(self, x, y):
        if self.model_name == 'slowfast':
            x = [i for i in x]
        elif self.model_name == 'vivit':
            x = x.permute(0,2,1,3,4)
        elif self.model_name =='vgg':
            x = x[:,:,0]

        out = self.model(x)
        loss =  self.loss_func(y, out)

        return loss, out

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

    def plot_cv(self, train_loss_log, valid_loss_log):
        plt.figure(figsize=(6.4,4.8))
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.plot(train_loss_log, label='train loss')
        plt.plot(valid_loss_log, label='test loss')
        plt.legend()
        plt.savefig(f"{self.output_dir}/learning_curves/learning_curve{str(self.fold)}.png")