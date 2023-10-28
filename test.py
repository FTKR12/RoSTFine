import sys
import os
from models import model_select
from runners.tester import Tester
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

def test(args):

    sperm_ids, grade_dict, video_dict, traj_dict = LoadData(args)
    kfold_train_dataloader, kfold_valid_dataloader = make_kfold_dataloader(
        args.kfold, 
        sperm_ids, video_dict, traj_dict, grade_dict, 
        num_frame=args.num_frame, 
        video_size=args.video_size, 
        batch_size=args.batch_size, 
        pathway=args.model_name
        )
    
    cv_result = []
    for fold in range(args.kfold):
        logger.info(f'Cross Validation {fold}')
        model = model_select(args)
        runner = Tester(
            args,
            fold, 
            kfold_valid_dataloader[fold],
            model
            )
        evaluated = runner.run()
        
        table = PrettyTable(field_names=evaluated.keys())
        table.add_row(list(evaluated.values()))
        logger.info(f'Cross Validation {fold}')
        logger.info(f'\n{table}')
        
        cv_result.append(evaluated)
    
    all_result = {}
    for k in cv_result[0]:
        all_result[k] = f"{str(np.mean([x[k] for x in cv_result]))} ± {str(np.std([x[k] for x in cv_result]))}"
    table = PrettyTable(field_names=all_result.keys())
    table.add_row(list(all_result.values()))
    logger.info(f'all result')
    logger.info(f'\n{table}')
    
if __name__ == "__main__":

    args = get_args()
    args.isTrain = False
    set_seed(args.seed)
    
    os.makedirs(args.load_dir+'/attn', exist_ok=True)

    # log settings
    logger = setup_logger('Sperm-Assessment', f'{args.load_dir}/logs', args.isTrain)
    logger.info(str(args).replace(',','\n'))

    # test
    test(args)