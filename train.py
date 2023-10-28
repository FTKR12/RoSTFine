import sys
import os
from models import model_select
from runners import runner_select
from dataloader import LoadData, make_kfold_dataloader
from prettytable import PrettyTable
import logging
import argparse
import time
import numpy as np
from utils import (
    get_args,
    setup_logger,
    set_seed,
    plot_all
)

def train(args):

    sperm_ids, grade_dict, video_dict, traj_dict = LoadData(args)
    kfold_train_dataloader, kfold_valid_dataloader = make_kfold_dataloader(
        args.kfold, 
        sperm_ids, video_dict, traj_dict, grade_dict, 
        num_frame=args.num_frame, 
        video_size=args.video_size, 
        batch_size=args.batch_size, 
        pathway=args.model_name
        )
    
    min_losses = []
    
    for fold in range(args.kfold):
        logger.info(f"Cross Validation {fold}")
        model = model_select(args)
        runner = runner_select(
            args,
            fold, 
            kfold_train_dataloader[fold],
            kfold_valid_dataloader[fold],
            model)
        runner.run()
        min_losses.append(runner.min_loss)
    
    table = PrettyTable(field_names=['cv1', 'cv2', 'cv3', 'cv4', 'cv5', 'avg', 'std'])
    table.add_row([min_losses[0], min_losses[1], min_losses[2], min_losses[3], min_losses[4], np.mean(min_losses), np.std(min_losses)])
    logger.info(f'\n{table}')

    plot_all(args)

if __name__ == "__main__":

    args = get_args()
    args.isTrain = True
    set_seed(args.seed)

    # make directories
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = f'{args.output_dir}/{args.model_name}'
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = f'{args.output_dir}/{cur_time}'
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+'/checkpoint', exist_ok=True)
    os.makedirs(args.output_dir+'/learning_curves', exist_ok=True)
    os.makedirs(args.output_dir+'/loss', exist_ok=True)
    os.makedirs(args.output_dir+'/logs', exist_ok=True)
    os.makedirs(args.output_dir+'/samples', exist_ok=True)
    os.makedirs(args.output_dir+'/tensorboard', exist_ok=True)

    # log settings
    logger = setup_logger('Sperm-Assessment', f'{args.output_dir}/logs', args.isTrain)
    logger.info(str(args).replace(',','\n'))

    # train
    train(args)