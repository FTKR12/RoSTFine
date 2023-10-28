from .trainer import Trainer
from .trainer_rostfine import RoSTFineTrainer
from .tester import Tester

def runner_select(args, fold, train_dataloader, valid_dataloader, model):
    runner_factory = {
        'not rostfine': Trainer(
            args=args,
            fold=fold,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            model=model,
        ),
        'rostfine': RoSTFineTrainer(
            args=args,
            fold=fold,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            model=model,
        )
    }
    if args.isTrain == True:
        which_trainer = 'rostfine' if args.model_name=='rostfine' else 'not rostfine'
        return runner_factory[which_trainer]
    elif args.isTrain == False:
        return runner_factory['test']